#!/usr/bin/env python3
"""
Configurable training script for Kinyarwanda speech recognition using Whisper
Maintains exact original configuration with configurable data amounts
Usage: python train.py --config configs/baseline.yaml
"""

import argparse
import yaml
import os
import torch
import transformers
from dataclasses import dataclass, field
from typing import Union, List, Dict, Any
import string
import json
import datasets
import numpy as np
import evaluate
import salt.dataset
import salt.metrics
import salt.constants
from salt.utils import DataCollatorCTCWithPadding as dcwp
import huggingface_hub
import peft
import pandas as pd
import tqdm.notebook as tqdm
import jiwer
from datetime import datetime
import logging
import importlib.metadata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Main configuration class"""

    experiment_name: str
    run_training: bool = True  # Set to False for baseline evaluation only
    dataset_subset: str = (
        "train_cleaned"  # Which subset from evie-8/kinyarwanda-speech-hackathon to use
    )
    use_wandb: bool = False
    use_mlflow: bool = True
    seed: int = 42


# Original DataCollator from your notebook
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def get_base_yaml_config(config: ExperimentConfig):
    """Returns the exact original YAML config with configurable modifications"""

    # Determine which dataset split to use based on configuration
    if config.dataset_subset == "baseline":
        # For baseline evaluation, use validation data only
        train_datasets = []
    else:
        # Use specific hourly split from evie-8/kinyarwanda-speech-hackathon
        train_datasets = [
            {
                "path": "evie-8/kinyarwanda-speech-hackathon",
                "split": config.dataset_subset,
            }
        ]

    yaml_config = f"""
pretrained_model: openai/whisper-large-v3

num_workers: 4
use_peft: False
lora_config:
    r: 32
    lora_alpha: 64
    target_modules: ["q_proj", "v_proj"]
    lora_dropout: 0.05
    bias: "none"

training_args:
    output_dir: {config.experiment_name}
    per_device_train_batch_size: 16
    per_device_eval_batch_size: 16
    gradient_accumulation_steps: 2  # 2 if using 2 GPUS (to keep batch size of 64)
    learning_rate: 1.0e-5
    warmup_steps: 100
    max_steps: 10000
    gradient_checkpointing: True
    gradient_checkpointing_kwargs:
      use_reentrant: True
    fp16: True
    eval_strategy: steps
    predict_with_generate: True
    generation_max_length: 200
    save_steps: 1000
    eval_steps: 200 # TODO: change back to 200
    logging_steps: 200 # TODO: change back to 200
    load_best_model_at_end: True
    metric_for_best_model: loss
    greater_is_better: False
    push_to_hub: False
    hub_model_id: jq/{config.experiment_name}
    save_total_limit: 2
    
train:
    download_datasets_in_parallel: True
    huggingface_load: {train_datasets}
    source:
      type: speech
      language: [kin]
      preprocessing:
        # Downsample some examples to 8KHz (to simulate phone audio) 
        - set_sample_rate:
            rate: 8_000
            p: 0.05
        # Then upsample again
        - set_sample_rate:
            rate: 16_000
        - normalize_audio
        - augment_audio_speed:
            p: 0.2
            low: 0.95
            high: 1.15
        - augment_audio_noise:
            max_relative_amplitude: 0.3    
    target:
      type: text
      preprocessing:
        - lower_case
        - clean_and_remove_punctuation:
            allowed_punctuation: "'"
      language: [kin]
    shuffle: True
validation:
    huggingface_load:
        # Held-out challenge data for validation
        - path: jq/kinyarwanda-speech-hackathon
          split: dev_test[:200]
    source:
      type: speech
      language: [kin]
      preprocessing:
        - set_sample_rate:
            rate: 16_000
    target:
      type: text
      language: [kin]
      preprocessing:
        - lower_case
        - clean_and_remove_punctuation:
            allowed_punctuation: "'"
"""
    return yaml_config


def setup_logging_backends(use_wandb, use_mlflow, experiment_name):
    """Setup logging backends exactly as in original"""

    installed = [dist.metadata["Name"] for dist in importlib.metadata.distributions()]

    if use_wandb:
        import wandb

        os.environ["WANDB_LOG_MODEL"] = "True"
        os.environ["WANDB_WATCH"] = "all"
        os.environ["WANDB_NOTEBOOK_NAME"] = "whisper_configurable.py"
        wandb.login()

    if use_mlflow:
        if "mlflow" not in installed:
            raise ImportError(
                "MLflow not installed. Install with: pip install mlflow psutil pynvml"
            )

        import mlflow
        import mlflow.pytorch
        from mlflow import MlflowClient
        from getpass import getpass

        # Set MLflow tracking credentials
        if "MLFLOW_TRACKING_USERNAME" not in os.environ:
            MLFLOW_TRACKING_USERNAME = getpass("Enter the MLFLOW_TRACKING_USERNAME: ")
            os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME

        if "MLFLOW_TRACKING_PASSWORD" not in os.environ:
            MLFLOW_TRACKING_PASSWORD = getpass("Enter the MLFLOW_TRACKING_PASSWORD: ")
            os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

        # Use consistent experiment name for all runs
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "kinyarwanda-asr-whisper-large-v3"

        # Set the MLflow tracking URI
        mlflow.set_tracking_uri("https://mlflow-sunbird-ce0ecfc14244.herokuapp.com/")
        mlflow.system_metrics.enable_system_metrics_logging()

        # Start a new run with the experiment name as the run name
        mlflow.start_run(run_name=experiment_name)


def prepare_dataset(
    example, sentence_to_prompt, feature_extractor, processor, p_prompt=0.0
):
    """Original prepare_dataset function"""
    audio = example["source"]
    input_features = feature_extractor(
        audio, sampling_rate=16000, device="cuda", do_normalize=True
    ).input_features[0]

    # Encode target text to label ids
    labels = processor.tokenizer(str(example["target"])).input_ids

    # Insert the language ID token into the second position of the sequence.
    language_id_tokens = salt.constants.SALT_LANGUAGE_TOKENS_WHISPER
    labels.insert(1, language_id_tokens[example["target.language"]])

    # If a prompt is known for a particular sentence, add it to the
    # training example with probability `p_prompt`.
    prompt = sentence_to_prompt.get(example["target"], None)
    if prompt:
        if np.random.random() < p_prompt:
            prompt_ids = list(processor.get_prompt_ids(prompt))
            labels = prompt_ids + labels

    # Create a new dictionary with the processed data
    processed_example = {
        "input_features": input_features,
        "labels": np.array(labels),
        "source.language": example["source.language"],
        "target.language": example["target.language"],
    }

    return processed_example


def run_baseline_evaluation(config_dict, experiment_config):
    """Run baseline evaluation without training"""
    logger.info("Running baseline evaluation (no training)")

    # Create validation dataset only
    valid_ds = salt.dataset.create(config_dict["validation"])

    # Setup model and processor
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        config_dict["pretrained_model"]
    )
    processor = transformers.WhisperProcessor.from_pretrained(
        config_dict["pretrained_model"], language=None, task="transcribe"
    )
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        config_dict["pretrained_model"]
    )

    # Model configuration
    model.config.suppress_tokens = []
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    # Create empty sentence_to_prompt for baseline
    sentence_to_prompt = {}

    # Prepare validation dataset
    val_data = valid_ds.map(
        lambda x: prepare_dataset(x, sentence_to_prompt, feature_extractor, processor),
        remove_columns=["source", "target"],
    )

    # Setup data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id
    )

    # Setup compute metrics
    compute_metrics = salt.metrics.multilingual_eval_fn(
        valid_ds,
        [evaluate.load("wer"), evaluate.load("cer")],
        processor.tokenizer,
        log_first_N_predictions=3,
        speech_processor=processor,
    )

    # Setup training arguments for evaluation
    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=experiment_config.experiment_name,
        per_device_eval_batch_size=config_dict["training_args"][
            "per_device_eval_batch_size"
        ],
        report_to=[
            platform
            for platform, use in [
                ("wandb", experiment_config.use_wandb),
                ("mlflow", experiment_config.use_mlflow),
            ]
            if use
        ],
    )

    # Create trainer
    trainer = transformers.Seq2SeqTrainer(
        args=training_args,
        model=model,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    # Run evaluation
    results = trainer.evaluate()

    if experiment_config.use_mlflow:
        import mlflow

        mlflow.log_params(config_dict)
        # Log additional experiment metadata
        mlflow.log_param("experiment_type", "baseline_evaluation")
        mlflow.log_param("dataset_subset", experiment_config.dataset_subset)
        mlflow.log_param("training_enabled", False)
        # Log results as metrics
        for key, value in results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

    return results


def run_full_training(config_dict, experiment_config):
    """Run the full training process exactly as in original"""
    logger.info(
        f"Starting training with dataset subset: {experiment_config.dataset_subset}"
    )

    # Create datasets directly (config already has correct dataset splits)
    train_ds = salt.dataset.create(config_dict["train"], verbose=True)
    valid_ds = salt.dataset.create(config_dict["validation"])

    # Pre-load datasets for prompts (exactly as in original)
    ds = datasets.load_dataset(
        "jq/kinyarwanda-speech-hackathon", split="train", num_proc=10
    )
    ds_filtered = datasets.load_dataset(
        "evie-8/kinyarwanda-speech-hackathon", split="train", num_proc=10
    )

    # Setup model and processor
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        config_dict["pretrained_model"]
    )
    processor = transformers.WhisperProcessor.from_pretrained(
        config_dict["pretrained_model"], language=None, task="transcribe"
    )
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        config_dict["pretrained_model"]
    )

    # Read in prompts exactly as in original
    text = list(ds["text"])
    prompts = list(ds["prompt"])
    sentence_to_prompt = {}
    for t, p in zip(text, prompts):
        sentence_to_prompt[t] = p

    # Prepare datasets
    train_data = train_ds.map(
        lambda x: prepare_dataset(x, sentence_to_prompt, feature_extractor, processor),
        remove_columns=["source", "target"],
    )
    val_data = valid_ds.map(
        lambda x: prepare_dataset(x, sentence_to_prompt, feature_extractor, processor),
        remove_columns=["source", "target"],
    )

    # Setup compute metrics
    compute_metrics = salt.metrics.multilingual_eval_fn(
        valid_ds,
        [evaluate.load("wer"), evaluate.load("cer")],
        processor.tokenizer,
        log_first_N_predictions=3,
        speech_processor=processor,
    )

    # Model configuration
    model.config.suppress_tokens = []
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    # PEFT setup if enabled
    if config_dict["use_peft"]:
        model = peft.prepare_model_for_kbit_training(model)
        lora_config = peft.LoraConfig(**config_dict["lora_config"])
        model.enable_input_require_grads()
        model = peft.get_peft_model(model, lora_config)
        model.config.use_cache = False
        model.print_trainable_parameters()

    # Setup data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id
    )

    # Launch training exactly as in original
    training_args = transformers.Seq2SeqTrainingArguments(
        **config_dict["training_args"],
        report_to=[
            platform
            for platform, use in [
                ("wandb", experiment_config.use_wandb),
                ("mlflow", experiment_config.use_mlflow),
            ]
            if use
        ],
    )

    trainer = transformers.Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    trainer.train()
    results = trainer.evaluate()

    # Log config settings for reference
    if experiment_config.use_mlflow:
        import mlflow

        mlflow.log_params(config_dict)
        # Log additional experiment metadata
        mlflow.log_param("experiment_type", "fine_tuning")
        mlflow.log_param("dataset_subset", experiment_config.dataset_subset)
        mlflow.log_param("training_enabled", True)
        # Log results as metrics
        for key, value in results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

    # Push to hub if enabled
    if config_dict["training_args"]["push_to_hub"]:
        processor.push_to_hub(
            config_dict["training_args"]["hub_model_id"], private=True
        )
        model.push_to_hub(config_dict["training_args"]["hub_model_id"], private=True)

    return results


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML file"""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return ExperimentConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Train Whisper on Kinyarwanda speech data"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file",
    )

    args = parser.parse_args()

    # Load experiment configuration
    experiment_config = load_experiment_config(args.config)

    # Set random seed
    torch.manual_seed(experiment_config.seed)
    np.random.seed(experiment_config.seed)

    logger.info(f"Starting experiment: {experiment_config.experiment_name}")
    logger.info(f"Configuration loaded from: {args.config}")

    # Setup logging backends
    setup_logging_backends(
        experiment_config.use_wandb,
        experiment_config.use_mlflow,
        experiment_config.experiment_name,
    )

    # Get base YAML config
    yaml_config_str = get_base_yaml_config(experiment_config)
    config_dict = yaml.safe_load(yaml_config_str)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_config.experiment_name = (
        f"{experiment_config.experiment_name}_{timestamp}"
    )
    config_dict["training_args"]["output_dir"] = experiment_config.experiment_name
    config_dict["training_args"]["hub_model_id"] = (
        f"jq/{experiment_config.experiment_name}"
    )

    os.makedirs(experiment_config.experiment_name, exist_ok=True)

    # Save config to output directory
    with open(
        os.path.join(experiment_config.experiment_name, "experiment_config.yaml"), "w"
    ) as f:
        yaml.dump(experiment_config.__dict__, f, default_flow_style=False)

    with open(
        os.path.join(experiment_config.experiment_name, "full_config.yaml"), "w"
    ) as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Run training or evaluation
    if experiment_config.run_training:
        results = run_full_training(config_dict, experiment_config)
    else:
        results = run_baseline_evaluation(config_dict, experiment_config)

    # Log results
    logger.info("=" * 50)
    logger.info("FINAL RESULTS:")
    for key, value in results.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 50)

    # Save results
    results_path = os.path.join(experiment_config.experiment_name, "results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f)

    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Experiment completed: {experiment_config.experiment_name}")

    # End MLflow run
    if experiment_config.use_mlflow:
        import mlflow

        mlflow.end_run()


if __name__ == "__main__":
    main()

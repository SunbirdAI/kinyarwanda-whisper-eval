#!/usr/bin/env python3
"""
Kinyarwanda Whisper fine-tuning script.
Usage: uv run python train.py --config configs/train_50h.yaml
"""

import argparse
import yaml
import os
import torch
import transformers
import numpy as np
import evaluate
import salt.dataset
import salt.metrics
import salt.constants
from datasets import load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    experiment_name: str
    dataset_subset: str = "audio_50h"
    use_wandb: bool = False
    use_mlflow: bool = True
    seed: int = 42


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def setup_logging_backends(use_wandb: bool, use_mlflow: bool, experiment_name: str):
    """Setup logging backends """
    if use_wandb:
        try:
            import wandb

            os.environ["WANDB_LOG_MODEL"] = "True"
            os.environ["WANDB_WATCH"] = "all"
            wandb.login()
            logger.info("✅ Weights & Biases initialized")
        except ImportError:
            logger.error("❌ wandb not installed")
            raise

    if use_mlflow:
        try:
            import mlflow
            import mlflow.pytorch
            from getpass import getpass

            if "MLFLOW_TRACKING_USERNAME" not in os.environ:
                os.environ["MLFLOW_TRACKING_USERNAME"] = getpass("MLFLOW username: ")
            if "MLFLOW_TRACKING_PASSWORD" not in os.environ:
                os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass("MLFLOW password: ")

            os.environ["MLFLOW_EXPERIMENT_NAME"] = "whisper-kinyarwanda-eval"
            mlflow.set_tracking_uri(
                "https://mlflow-sunbird-ce0ecfc14244.herokuapp.com/"
            )
            mlflow.system_metrics.enable_system_metrics_logging()
            mlflow.start_run(run_name=experiment_name)
            logger.info("✅ MLflow initialized")
        except ImportError:
            logger.error("❌ mlflow not installed")
            raise


def prepare_dataset(
    example, sentence_to_prompt, feature_extractor, processor, p_prompt=0.0
):
    """Data prep """
    audio = example["source"]
    input_features = feature_extractor(
        audio,
        sampling_rate=16000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        do_normalize=True,
    ).input_features[0]

    # Encode target text to label ids
    labels = processor.tokenizer(str(example["target"])).input_ids

    # Insert the language ID token into the second position of the sequence.
    labels.insert(
        1, salt.constants.SALT_LANGUAGE_TOKENS_WHISPER[example["target.language"]]
    )

    # If a prompt is known for a particular sentence, add it to the training example
    prompt = sentence_to_prompt.get(example["target"], None)
    if prompt and np.random.random() < p_prompt:
        prompt_ids = list(processor.get_prompt_ids(prompt))
        labels = prompt_ids + labels

    return {
        "input_features": input_features,
        "labels": np.array(labels),
        "source.language": example["source.language"],
        "target.language": example["target.language"],
    }


def build_salt_config(experiment_config: ExperimentConfig) -> dict:
    """Build the SALT config"""

    # Create YAML structure 
    config_yaml = f"""
pretrained_model: openai/whisper-large-v3
num_workers: 4
use_peft: false

training_args:
    output_dir: {experiment_config.experiment_name}
    per_device_train_batch_size: 16
    per_device_eval_batch_size: 16
    gradient_accumulation_steps: 2
    learning_rate: 1.0e-5
    warmup_steps: 100
    max_steps: 10000
    gradient_checkpointing: true
    gradient_checkpointing_kwargs:
      use_reentrant: false
    fp16: true
    eval_strategy: steps
    predict_with_generate: true
    generation_max_length: 200
    save_steps: 1000
    eval_steps: 200
    logging_steps: 200
    load_best_model_at_end: true
    metric_for_best_model: loss
    greater_is_better: false
    push_to_hub: true
    hub_model_id: akera/{experiment_config.experiment_name}
    save_total_limit: 2

train:
    download_datasets_in_parallel: false
    huggingface_load:
        - path: evie-8/kinyarwanda-speech-hackathon
          name: {experiment_config.dataset_subset}
          split: train
    source:
      type: speech
      language: [kin]
      preprocessing:
        # preprocessing from SALT
        - set_sample_rate:
            rate: 8_000
            p: 0.05
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
    shuffle: false

validation:
    huggingface_load:
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

    return yaml.safe_load(config_yaml)


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Load experiment config from YAML."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return ExperimentConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Train Whisper on Kinyarwanda speech data"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment configuration"
    )
    args = parser.parse_args()

    # Load configuration
    experiment_config = load_experiment_config(args.config)
    torch.manual_seed(experiment_config.seed)
    np.random.seed(experiment_config.seed)

    logger.info("=" * 60)
    logger.info(f"🎯 Experiment: {experiment_config.experiment_name}")
    logger.info(f"📊 Dataset: {experiment_config.dataset_subset}")
    logger.info(f"🎲 Seed: {experiment_config.seed}")
    logger.info("=" * 60)

    # Setup logging
    setup_logging_backends(
        experiment_config.use_wandb,
        experiment_config.use_mlflow,
        experiment_config.experiment_name,
    )

    # Build SALT config
    config = build_salt_config(experiment_config)

    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{experiment_config.experiment_name}_{timestamp}"
    config["training_args"]["output_dir"] = output_dir
    config["training_args"][
        "hub_model_id"
    ] = f"akera/{experiment_config.experiment_name}_{timestamp}"

    # Create output directory and save configs
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "experiment_config.yaml"), "w") as f:
        yaml.dump(experiment_config.__dict__, f, default_flow_style=False)
    with open(os.path.join(output_dir, "full_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info("📊 Creating datasets using SALT...")

    # Create datasets using SALT
    train_ds = salt.dataset.create(config["train"], verbose=True)
    valid_ds = salt.dataset.create(config["validation"])

    logger.info("📚 Loading prompts dataset...")
    # Load prompts
    try:
        ds_prompts = load_dataset(
            "evie-8/kinyarwanda-speech-hackathon",
            name=experiment_config.dataset_subset,
            split="train",
        )
        text = list(ds_prompts["text"])
        prompts = list(ds_prompts["prompt"])
        sentence_to_prompt = {t: p for t, p in zip(text, prompts)}
        logger.info(f"✅ Loaded {len(sentence_to_prompt)} prompts")
    except Exception as e:
        logger.warning(f"⚠️ Could not load prompts: {e}")
        sentence_to_prompt = {}

    logger.info("🤖 Loading model and processor...")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        config["pretrained_model"]
    )
    processor = transformers.WhisperProcessor.from_pretrained(
        config["pretrained_model"], language=None, task="transcribe"
    )
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        config["pretrained_model"]
    )

    logger.info("🔧 Preparing datasets...")
    # Map datasets
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

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id
    )

    # Training arguments
    training_args = transformers.Seq2SeqTrainingArguments(
        **config["training_args"],
        disable_tqdm=False,
        report_to=[
            platform
            for platform, use in [
                ("wandb", experiment_config.use_wandb),
                ("mlflow", experiment_config.use_mlflow),
            ]
            if use
        ],
    )

    # Trainer setup
    trainer = transformers.Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    logger.info("🏃 Starting training...")
    trainer.train()

    logger.info("📝 Running final evaluation...")
    results = trainer.evaluate()

    # Log to MLflow if enabled
    if experiment_config.use_mlflow:
        import mlflow

        mlflow.log_params(config)
        mlflow.log_param("experiment_type", "fine_tuning")
        mlflow.log_param("dataset_subset", experiment_config.dataset_subset)
        for key, value in results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        mlflow.end_run()

    # Save results
    results_path = os.path.join(output_dir, "results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f)

    logger.info("=" * 60)
    logger.info("🎉 FINAL RESULTS:")
    for key, value in results.items():
        logger.info(f"   {key}: {value}")
    logger.info("=" * 60)
    logger.info(f"💾 Results saved to: {results_path}")
    logger.info(f"✅ Experiment completed: {experiment_config.experiment_name}")


if __name__ == "__main__":
    main()

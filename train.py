#!/usr/bin/env python3
"""
Multi-language Whisper fine-tuning script.
Usage: uv run python train.py --config configs/train_zulu_100h.yaml
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
from typing import Any, Dict, List, Union, Optional
from datetime import datetime
import logging
from transformers import EarlyStoppingCallback


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class LanguageConfig:
    """Language-specific configuration"""

    code: str  # e.g., 'zul', 'kin', 'en'
    name: str  # e.g., 'Zulu', 'Kinyarwanda', 'English'
    train_dataset_path: str
    validation_dataset_path: str
    train_dataset_name: Optional[str] = None
    train_split: str = "train"
    validation_dataset_name: Optional[str] = None
    validation_split: str = "dev"
    insert_language_token: bool = True
    custom_language_token: Optional[int] = None


@dataclass
class ExperimentConfig:
    experiment_name: str
    language: LanguageConfig
    dataset_subset: str = "audio_50h"
    use_wandb: bool = False
    use_mlflow: bool = True
    seed: int = 42
    mlflow_experiment_base_name: str = "whisper-zul-eval"


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
        # batch["input_features"] = batch["input_features"].to(torch.bfloat16)

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def setup_logging_backends(
    use_wandb: bool, use_mlflow: bool, experiment_name: str, mlflow_experiment_name: str
):
    """Setup logging backends"""
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

            mlflow.set_tracking_uri(
                "https://mlflow-sunbird-ce0ecfc14244.herokuapp.com/"
            )
            mlflow.set_experiment(mlflow_experiment_name)
            mlflow.system_metrics.enable_system_metrics_logging()
            mlflow.start_run(run_name=experiment_name)
            logger.info("✅ MLflow initialized")
        except ImportError:
            logger.error("❌ mlflow not installed")
            raise


def prepare_dataset(
    example,
    sentence_to_prompt,
    feature_extractor,
    processor,
    language_config,
    p_prompt=0.0,
):
    """Data prep"""
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
    if example["target.language"] in salt.constants.SALT_LANGUAGE_TOKENS_WHISPER:
        labels.insert(
            1, salt.constants.SALT_LANGUAGE_TOKENS_WHISPER[example["target.language"]]
        )
    # If language not in SALT constants, skip language token insertion

    # If a prompt is known for a particular sentence, add it to the training example
    prompt = sentence_to_prompt.get(example["target"], None)
    if prompt and np.random.random() < p_prompt:
        prompt_ids = list(processor.get_prompt_ids(prompt))
        labels = prompt_ids + labels

    # Truncate labels if they're too long (Whisper max is 448 tokens)
    if len(labels) > 448:
        labels = labels[:448]

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
num_workers: 12
use_peft: false

training_args:
    output_dir: {experiment_config.experiment_name}
    per_device_train_batch_size: 16
    per_device_eval_batch_size: 16
    dataloader_pin_memory: true
    gradient_accumulation_steps: 2
    learning_rate: 1.0e-5
    warmup_steps: 100
    max_steps: 25000
    gradient_checkpointing: true
    gradient_checkpointing_kwargs:
      use_reentrant: false
    fp16: true
    eval_strategy: steps
    predict_with_generate: true
    generation_max_length: 200
    save_steps: 400
    eval_steps: 400
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
        - path: {experiment_config.language.train_dataset_path}
          {"name: " + experiment_config.language.train_dataset_name if experiment_config.language.train_dataset_name else ""}
          split: {experiment_config.language.train_split}
          num_proc: 10
        - path: {experiment_config.language.train_dataset_path}
          {"name: " + experiment_config.language.train_dataset_name if experiment_config.language.train_dataset_name else ""}
          split: {experiment_config.language.train_split}[:100]
          num_proc: 10
    source:
      type: speech
      language: [{experiment_config.language.code}]
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
      language: [{experiment_config.language.code}]
    shuffle: True

validation:
    huggingface_load:
        - path: {experiment_config.language.validation_dataset_path}
          {"name: " + experiment_config.language.validation_dataset_name if experiment_config.language.validation_dataset_name else ""}
          split: {experiment_config.language.validation_split}
    source:
      type: speech
      language: [{experiment_config.language.code}]
      preprocessing:
        - set_sample_rate:
            rate: 16_000
    target:
      type: text
      language: [{experiment_config.language.code}]
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

    # Create LanguageConfig from nested dict
    language_config = LanguageConfig(**config_dict.pop("language"))
    config_dict["language"] = language_config

    return ExperimentConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Train Whisper on multilingual speech data"
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
    logger.info(
        f"🌍 Language: {experiment_config.language.name} ({experiment_config.language.code})"
    )
    logger.info(f"📊 Dataset: {experiment_config.dataset_subset}")
    logger.info(f"🎲 Seed: {experiment_config.seed}")
    logger.info("=" * 60)

    # Create MLflow experiment name
    mlflow_experiment_name = experiment_config.mlflow_experiment_base_name

    # Setup logging
    setup_logging_backends(
        experiment_config.use_wandb,
        experiment_config.use_mlflow,
        experiment_config.experiment_name,
        mlflow_experiment_name,
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
        # Convert dataclasses to dict for YAML serialization
        config_dict = {
            "experiment_name": experiment_config.experiment_name,
            "language": {
                "code": experiment_config.language.code,
                "name": experiment_config.language.name,
                "train_dataset_path": experiment_config.language.train_dataset_path,
                "train_dataset_name": experiment_config.language.train_dataset_name,
                "train_split": experiment_config.language.train_split,
                "validation_dataset_path": experiment_config.language.validation_dataset_path,
                "validation_dataset_name": experiment_config.language.validation_dataset_name,
                "validation_split": experiment_config.language.validation_split,
            },
            "dataset_subset": experiment_config.dataset_subset,
            "use_wandb": experiment_config.use_wandb,
            "use_mlflow": experiment_config.use_mlflow,
            "seed": experiment_config.seed,
            "mlflow_experiment_base_name": experiment_config.mlflow_experiment_base_name,
        }
        yaml.dump(config_dict, f, default_flow_style=False)
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
            experiment_config.language.train_dataset_path,
            name=experiment_config.language.train_dataset_name,
            split=experiment_config.language.train_split,
        )
        # Check if prompts exist in the dataset
        if "prompt" in ds_prompts.column_names and "text" in ds_prompts.column_names:
            text = list(ds_prompts["text"])
            prompts = list(ds_prompts["prompt"])
            sentence_to_prompt = {t: p for t, p in zip(text, prompts)}
            logger.info(f"✅ Loaded {len(sentence_to_prompt)} prompts")
        else:
            logger.info(
                "ℹ️ No prompt column found in dataset, proceeding without prompts"
            )
            sentence_to_prompt = {}
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
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2"
    )

    logger.info("🔧 Preparing datasets...")

    # Map datasets
    train_data = train_ds.map(
        lambda x: prepare_dataset(
            x,
            sentence_to_prompt,
            feature_extractor,
            processor,
            experiment_config.language,
        ),
        remove_columns=["source", "target"],
    )
    val_data = valid_ds.map(
        lambda x: prepare_dataset(
            x,
            sentence_to_prompt,
            feature_extractor,
            processor,
            experiment_config.language,
        ),
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        processing_class=processor,
    )

    # log GPU name and VRAM to mlflow
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

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
        mlflow.log_param("language_code", experiment_config.language.code)
        mlflow.log_param("language_name", experiment_config.language.name)
        mlflow.log_param("gpu_name", gpu_name)
        mlflow.log_param("vram_gb", vram_gb)
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

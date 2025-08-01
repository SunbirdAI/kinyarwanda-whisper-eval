#!/usr/bin/env python3
"""
Multi-language Whisper fine-tuning script with dynamic hour-based splits.
Creates DatasetDict splits and loads with SALT - exactly like Kinyarwanda approach.

Usage: uv run python train.py --config configs/sna/baseline.yaml
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
from transformers import EarlyStoppingCallback
from data_utils import create_and_save_hour_based_splits, get_language_code_mapping

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    experiment_name: str
    dataset_name: str = "evie-8/afrivoices"
    dataset_subset: str = "sna"  # sna, ful, lin
    language_code: str = "sna"
    target_hours: float = 1.0
    use_wandb: bool = False
    use_mlflow: bool = True
    seed: int = 42
    run_training: bool = True


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
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)

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
    """Setup logging backends"""
    if use_mlflow:
        try:
            import mlflow
            import mlflow.pytorch
            from getpass import getpass

            if "MLFLOW_TRACKING_USERNAME" not in os.environ:
                os.environ["MLFLOW_TRACKING_USERNAME"] = getpass("MLFLOW username: ")
            if "MLFLOW_TRACKING_PASSWORD" not in os.environ:
                os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass("MLFLOW password: ")

            os.environ["MLFLOW_EXPERIMENT_NAME"] = "whisper-multilingual-eval"
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
    example,
    sentence_to_prompt,
    feature_extractor,
    processor,
    language_code,
    p_prompt=0.0,
):
    """Data prep with language-specific handling"""
    try:
        audio = example["source"]
        input_features = feature_extractor(
            audio,
            sampling_rate=16000,
            device="cuda" if torch.cuda.is_available() else "cpu",
            do_normalize=True,
        ).input_features[0]

        # Encode target text to label ids
        labels = processor.tokenizer(str(example["target"])).input_ids

        # Insert the language ID token - check if language code exists in SALT constants
        if language_code in salt.constants.SALT_LANGUAGE_TOKENS_WHISPER:
            language_token = salt.constants.SALT_LANGUAGE_TOKENS_WHISPER[language_code]
            labels.insert(1, language_token)
        else:
            logger.warning(f"Language code {language_code} not found in SALT constants")

        # If a prompt is known for a particular sentence, add it to the training example
        prompt = sentence_to_prompt.get(example["target"], None)
        if prompt and np.random.random() < p_prompt:
            prompt_ids = list(processor.get_prompt_ids(prompt))
            labels = prompt_ids + labels

        return {
            "input_features": input_features,
            "labels": np.array(labels),
            "source.language": example.get("source.language", language_code),
            "target.language": example.get("target.language", language_code),
        }
    except Exception as e:
        logger.error(f"Error processing example: {e}")
        logger.error(
            f"Example keys: {example.keys() if hasattr(example, 'keys') else 'No keys'}"
        )
        # Return a minimal valid example to avoid None
        return {
            "input_features": np.zeros((80, 3000)),  # Minimal audio features
            "labels": np.array([processor.tokenizer.eos_token_id]),
            "source.language": language_code,
            "target.language": language_code,
        }


def build_salt_config(
    experiment_config: ExperimentConfig, dataset_path: str, has_train_data: bool = True
) -> dict:
    """Build the SALT config with path to saved DatasetDict - exactly like Kinyarwanda approach"""

    # Only include train config if we have training data
    if has_train_data:
        train_config = f"""
train:
    download_datasets_in_parallel: false
    huggingface_load:
        - path: {dataset_path}
          split: train
          num_proc: 10
    source:
      type: speech
      language: [{experiment_config.language_code}]
      column: audio
      preprocessing:
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
      column: text
      preprocessing:
        - lower_case
        - clean_and_remove_punctuation:
            allowed_punctuation: "'"
      language: [{experiment_config.language_code}]
    shuffle: True
"""
    else:
        train_config = ""

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
    gradient_checkpointing: false
    gradient_checkpointing_kwargs:
      use_reentrant: false
    bf16: true
    eval_strategy: steps
    predict_with_generate: true
    generation_max_length: 200
    save_steps: 400
    eval_steps: 200
    logging_steps: 200
    load_best_model_at_end: true
    metric_for_best_model: loss
    greater_is_better: false
    push_to_hub: true
    hub_model_id: akera/{experiment_config.experiment_name}
    save_total_limit: 2
{train_config}
validation:
    huggingface_load:
        - path: {dataset_path}
          split: test
    source:
      type: speech
      language: [{experiment_config.language_code}]
      column: audio
      preprocessing:
        - set_sample_rate:
            rate: 16_000
    target:
      type: text
      language: [{experiment_config.language_code}]
      column: text
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
        description="Train Whisper on multi-language speech data"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment configuration"
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of cached splits",
    )
    args = parser.parse_args()

    # Load configuration
    experiment_config = load_experiment_config(args.config)
    torch.manual_seed(experiment_config.seed)
    np.random.seed(experiment_config.seed)

    logger.info("=" * 60)
    logger.info(f"🎯 Experiment: {experiment_config.experiment_name}")
    logger.info(f"🌍 Language: {experiment_config.language_code}")
    logger.info(
        f"📊 Dataset: {experiment_config.dataset_name}/{experiment_config.dataset_subset}"
    )
    if experiment_config.run_training:
        logger.info(f"⏰ Target hours: {experiment_config.target_hours}")
    else:
        logger.info("📝 Mode: Baseline evaluation (no training)")
    logger.info(f"🎲 Seed: {experiment_config.seed}")
    logger.info("=" * 60)

    # Create and save hour-based splits as DatasetDict (like Kinyarwanda pre-existing subsets)
    if experiment_config.run_training:
        logger.info("🔄 Creating and saving hour-based splits...")
        train_split_info, test_split_info = create_and_save_hour_based_splits(
            dataset_name=experiment_config.dataset_name,
            language_subset=experiment_config.dataset_subset,
            target_hours=experiment_config.target_hours,
            test_size=300,
            seed=experiment_config.seed,
            force_recreate=args.force_recreate,
        )
    else:
        # For baseline evaluation, create empty training split and normal test split
        logger.info("🔄 Creating test split for baseline evaluation...")
        train_split_info, test_split_info = create_and_save_hour_based_splits(
            dataset_name=experiment_config.dataset_name,
            language_subset=experiment_config.dataset_subset,
            target_hours=0.0,  # Empty training split
            test_size=300,
            seed=experiment_config.seed,
            force_recreate=args.force_recreate,
        )

    if train_split_info["is_cached"]:
        logger.info(
            f"✅ Using cached splits: {train_split_info['num_samples']} train, {test_split_info['num_samples']} test"
        )
    else:
        if experiment_config.run_training:
            logger.info(
                f"✅ Created new splits: {train_split_info['num_samples']} train ({train_split_info['actual_hours']:.2f}h), {test_split_info['num_samples']} test"
            )
        else:
            logger.info(
                f"✅ Created baseline test split: {test_split_info['num_samples']} samples"
            )

    # Setup logging
    setup_logging_backends(
        experiment_config.use_wandb,
        experiment_config.use_mlflow,
        experiment_config.experiment_name,
    )

    # Build SALT config with path to saved DatasetDict (exactly like Kinyarwanda)
    has_train_data = train_split_info["num_samples"] > 0
    config = build_salt_config(
        experiment_config, train_split_info["dataset_path"], has_train_data
    )

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
    with open(os.path.join(output_dir, "split_info.yaml"), "w") as f:
        yaml.dump(
            {"train": train_split_info, "test": test_split_info},
            f,
            default_flow_style=False,
        )

    # Convert to the format SALT would produce (source/target)
    def convert_to_salt_format(example):
        return {
            "source": example["audio"]["array"],
            "target": example["text"],
            "source.language": experiment_config.language_code,
            "target.language": experiment_config.language_code,
        }

    # Debug: Check what's actually in the saved DatasetDict
    logger.info("🔍 Checking saved DatasetDict structure...")
    try:
        from datasets import DatasetDict

        saved_dict = DatasetDict.load_from_disk(train_split_info["dataset_path"])
        logger.info(f"Available splits in DatasetDict: {list(saved_dict.keys())}")
        for split_name, split_data in saved_dict.items():
            logger.info(f"Split '{split_name}': {len(split_data)} samples")
            if len(split_data) > 0:
                logger.info(f"Sample from '{split_name}': {list(split_data[0].keys())}")
    except Exception as e:
        logger.error(f"Error loading DatasetDict: {e}")

    # Create datasets using SALT (normal IterableDatasets - exactly like Kinyarwanda)
    logger.info("📊 Creating datasets using SALT...")

    if experiment_config.run_training and has_train_data:
        train_ds = salt.dataset.create(config["train"], verbose=True)
        logger.info("✅ Training dataset created: IterableDataset")
    else:
        train_ds = None
        logger.info("✅ No training dataset (baseline mode or empty split)")

    # Create datasets - bypass SALT and load directly from our DatasetDict
    logger.info("📊 Creating datasets directly from saved DatasetDict...")
    from datasets import DatasetDict

    saved_dict = DatasetDict.load_from_disk(train_split_info["dataset_path"])

    if experiment_config.run_training and has_train_data:
        raw_train_ds = saved_dict["train"]
        logger.info(f"✅ Loaded {len(raw_train_ds)} training samples directly")

        train_ds = raw_train_ds.map(convert_to_salt_format)
        logger.info("✅ Training dataset converted to SALT format")
    else:
        train_ds = None
        logger.info("✅ No training dataset (baseline mode or empty split)")

    # Always load validation dataset directly
    raw_valid_ds = saved_dict["test"]
    logger.info(f"✅ Loaded {len(raw_valid_ds)} validation samples directly")
    valid_ds = raw_valid_ds.map(convert_to_salt_format)
    logger.info("✅ Validation dataset converted to SALT format")

    # No prompts for the new dataset - keep it simple
    sentence_to_prompt = {}
    logger.info("ℹ️ No prompts dataset for new languages - skipping prompt loading")

    logger.info("🤖 Loading model and processor...")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        config["pretrained_model"]
    )
    processor = transformers.WhisperProcessor.from_pretrained(
        config["pretrained_model"], language=None, task="transcribe"
    )
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        config["pretrained_model"], torch_dtype=torch.bfloat16
    )

    logger.info("🔧 Preparing datasets...")

    # Debug: Check what's in the validation dataset
    logger.info("🔍 Debugging validation dataset...")
    try:
        sample_count = 0
        for i, sample in enumerate(valid_ds):
            sample_count += 1
            if i == 0:
                logger.info(f"First sample keys: {sample.keys()}")
                logger.info(f"Sample structure: {type(sample)}")
            if i >= 2:  # Just check first 3 samples
                break
        logger.info(f"✅ Found {sample_count} samples in validation dataset")
    except Exception as e:
        logger.error(f"❌ Error iterating validation dataset: {e}")
        logger.error("Dataset might be empty or corrupted")

    # Map datasets with language-specific handling (normal processing - just like Kinyarwanda)
    if train_ds is not None:
        train_data = train_ds.map(
            lambda x: prepare_dataset(
                x,
                sentence_to_prompt,
                feature_extractor,
                processor,
                experiment_config.language_code,
            ),
            remove_columns=["source", "target"],
        )
    else:
        train_data = None

    logger.info("🔧 Mapping validation dataset...")
    val_data = valid_ds.map(
        lambda x: prepare_dataset(
            x,
            sentence_to_prompt,
            feature_extractor,
            processor,
            experiment_config.language_code,
        ),
        remove_columns=["source", "target"],
    )
    logger.info("✅ Validation dataset mapped")

    # Setup compute metrics - simple version that works with our data structure
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Decode predictions
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

        # Normalize text for evaluation
        normalized_preds = [pred.strip().lower() for pred in decoded_preds]
        normalized_labels = [label.strip().lower() for label in decoded_labels]

        # Compute WER and CER
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")

        wer = wer_metric.compute(
            predictions=normalized_preds, references=normalized_labels
        )
        cer = cer_metric.compute(
            predictions=normalized_preds, references=normalized_labels
        )

        return {"wer": wer, "cer": cer, "score": 1.0 - (0.6 * cer + 0.4 * wer)}

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
        train_dataset=train_data,  # Will be None for baseline evaluation
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=(
            [EarlyStoppingCallback(early_stopping_patience=10)]
            if experiment_config.run_training
            else []
        ),
        processing_class=processor,
    )

    # Log GPU info
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if torch.cuda.is_available()
        else 0
    )

    if experiment_config.run_training:
        logger.info("🏃 Starting training...")
        trainer.train()
        logger.info("📝 Running final evaluation...")
        results = trainer.evaluate()
    else:
        logger.info("📝 Running baseline evaluation (no training)...")
        results = trainer.evaluate()

    # Log to MLflow if enabled
    if experiment_config.use_mlflow:
        import mlflow

        mlflow.log_params(config)
        experiment_type = (
            "baseline_evaluation"
            if not experiment_config.run_training
            else "fine_tuning"
        )
        mlflow.log_param("experiment_type", experiment_type)
        mlflow.log_param("language", experiment_config.language_code)
        mlflow.log_param("dataset_subset", experiment_config.dataset_subset)
        mlflow.log_param("gpu_name", gpu_name)
        mlflow.log_param("vram_gb", vram_gb)

        if experiment_config.run_training:
            mlflow.log_param("target_hours", experiment_config.target_hours)
            mlflow.log_param("actual_hours", train_split_info.get("actual_hours", 0))
            mlflow.log_param("train_samples", train_split_info.get("num_samples", 0))
        else:
            mlflow.log_param("target_hours", 0)
            mlflow.log_param("actual_hours", 0)
            mlflow.log_param("train_samples", 0)

        mlflow.log_param("test_samples", test_split_info.get("num_samples", 0))

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

#!/usr/bin/env python3
"""
Configurable training script for Kinyarwanda speech recognition using Whisper.
Supports baseline evaluation and fine-tuning with different dataset sizes.

Evaluation settings:
- Uses dev_test[:300] for validation
- Uses num_beams=5 for generation
- Applies submission-style normalization (strip punctuation + lowercase)
- Calculates final score: 1 - (0.6 * CER + 0.4 * WER)

Usage: python train.py --config configs/baseline.yaml
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
import huggingface_hub
import peft
import jiwer
import string
from dataclasses import dataclass
from typing import Union, List, Dict, Any
from datetime import datetime
import logging
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def strip_punctuation(text):
    """Strip punctuation for evaluation normalization."""
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def normalise(texts):
    """Normalize texts for evaluation (lowercase + strip punctuation)."""
    return [strip_punctuation(t.lower()) for t in texts]


def create_evaluation_compute_metrics_fn(valid_ds, processor):
    """Create compute_metrics function with submission-style evaluation."""

    def compute_metrics(eval_preds):
        # Use SALT's multilingual evaluation first (for compatibility)
        salt_results = salt.metrics.multilingual_eval_fn(
            valid_ds,
            [evaluate.load("wer"), evaluate.load("cer")],
            processor.tokenizer,
            log_first_N_predictions=3,
            speech_processor=processor,
        )(eval_preds)

        # Extract predictions and labels for submission-style evaluation
        predictions = eval_preds.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        label_ids = eval_preds.label_ids

        # Decode predictions and labels
        predicted_texts = processor.batch_decode(predictions, skip_special_tokens=True)
        label_texts = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Apply normalization
        normalized_predictions = normalise(predicted_texts)
        normalized_labels = normalise(label_texts)

        # Calculate submission-style metrics
        final_wer = jiwer.wer(normalized_labels, normalized_predictions)
        final_cer = jiwer.cer(normalized_labels, normalized_predictions)
        final_score = 1 - (0.6 * final_cer + 0.4 * final_wer)

        # Add submission-style metrics to results
        salt_results.update(
            {
                "final_WER": final_wer,
                "final_CER": final_cer,
                "final_score": final_score,
            }
        )

        return salt_results

    return compute_metrics


@dataclass
class ExperimentConfig:
    """Experiment configuration dataclass."""

    experiment_name: str
    run_training: bool = True
    dataset_subset: str = "train_cleaned"
    use_wandb: bool = False
    use_mlflow: bool = True
    seed: int = 42


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text tasks with proper padding."""

    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Process audio inputs
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Process label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if already present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def setup_logging_backends(use_wandb: bool, use_mlflow: bool, experiment_name: str):
    """Setup MLflow and/or Weights & Biases logging."""
    if use_wandb:
        try:
            import wandb

            os.environ["WANDB_LOG_MODEL"] = "True"
            os.environ["WANDB_WATCH"] = "all"
            wandb.login()
            logger.info("✅ Weights & Biases initialized")
        except ImportError:
            logger.error("❌ wandb not installed. Install with: pip install wandb")
            raise

    if use_mlflow:
        try:
            import mlflow
            import mlflow.pytorch
            from getpass import getpass

            # Set up MLflow credentials
            if "MLFLOW_TRACKING_USERNAME" not in os.environ:
                username = getpass("Enter MLFLOW_TRACKING_USERNAME: ")
                os.environ["MLFLOW_TRACKING_USERNAME"] = username

            if "MLFLOW_TRACKING_PASSWORD" not in os.environ:
                password = getpass("Enter MLFLOW_TRACKING_PASSWORD: ")
                os.environ["MLFLOW_TRACKING_PASSWORD"] = password

            os.environ["MLFLOW_EXPERIMENT_NAME"] = "whisper-kinyarwanda-eval"
            mlflow.set_tracking_uri(
                "https://mlflow-sunbird-ce0ecfc14244.herokuapp.com/"
            )
            mlflow.system_metrics.enable_system_metrics_logging()
            mlflow.start_run(run_name=experiment_name)
            logger.info("✅ MLflow initialized")
        except ImportError:
            logger.error(
                "❌ MLflow not installed. Install with: pip install mlflow psutil pynvml"
            )
            raise


def prepare_dataset(
    example, sentence_to_prompt, feature_extractor, processor, p_prompt=0.0
):
    """Prepare dataset example for training/evaluation."""
    audio = example["source"]

    # Extract audio features
    input_features = feature_extractor(
        audio, sampling_rate=16000, device="cuda", do_normalize=True
    ).input_features[0]

    # Encode target text to label ids
    labels = processor.tokenizer(str(example["target"])).input_ids

    # Insert language ID token
    language_id_tokens = salt.constants.SALT_LANGUAGE_TOKENS_WHISPER
    labels.insert(1, language_id_tokens[example["target.language"]])

    # Add prompt with probability p_prompt
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


def get_training_config(experiment_config: ExperimentConfig) -> dict:
    """Generate training configuration based on experiment settings."""
    if experiment_config.dataset_subset == "baseline":
        train_datasets_yaml = " []"
    else:
        train_datasets_yaml = f"""
        - path: evie-8/kinyarwanda-speech-hackathon
          name: {experiment_config.dataset_subset}
          split: train"""

    config_template = f"""
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
    output_dir: {experiment_config.experiment_name}
    per_device_train_batch_size: 16
    per_device_eval_batch_size: 16
    gradient_accumulation_steps: 2
    learning_rate: 1.0e-5
    warmup_steps: 100
    max_steps: 10000
    gradient_checkpointing: True
    gradient_checkpointing_kwargs:
      use_reentrant: False
    fp16: True
    eval_strategy: steps
    predict_with_generate: True
    generation_max_length: 200
    generation_num_beams: 5
    save_steps: 1000
    eval_steps: 200
    logging_steps: 200
    load_best_model_at_end: True
    metric_for_best_model: loss
    greater_is_better: False
    push_to_hub: False
    hub_model_id: jq/{experiment_config.experiment_name}
    save_total_limit: 2
    
train:
    download_datasets_in_parallel: True
    huggingface_load:{train_datasets_yaml}
    source:
      type: speech
      language: [kin]
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
      preprocessing:
        - lower_case
        - clean_and_remove_punctuation:
            allowed_punctuation: "'"
      language: [kin]
    shuffle: True

validation:
    huggingface_load:
        - path: jq/kinyarwanda-speech-hackathon
          split: dev_test[:300]
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
    return yaml.safe_load(config_template)


def run_baseline_evaluation(config_dict: dict, experiment_config: ExperimentConfig):
    """Run baseline evaluation without training."""
    logger.info("🔍 Running baseline evaluation (no training)")

    logger.info("📊 Creating validation dataset...")
    valid_ds = salt.dataset.create(config_dict["validation"])

    logger.info("🤖 Loading model and processor...")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        config_dict["pretrained_model"]
    )
    processor = transformers.WhisperProcessor.from_pretrained(
        config_dict["pretrained_model"], language=None, task="transcribe"
    )
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        config_dict["pretrained_model"]
    )

    # Configure model
    model.config.suppress_tokens = []
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    sentence_to_prompt = {}

    logger.info("🔧 Preparing validation dataset...")
    val_data = valid_ds.map(
        lambda x: prepare_dataset(x, sentence_to_prompt, feature_extractor, processor),
        remove_columns=["source", "target"],
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id
    )

    compute_metrics = create_evaluation_compute_metrics_fn(valid_ds, processor)

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=experiment_config.experiment_name,
        per_device_eval_batch_size=config_dict["training_args"][
            "per_device_eval_batch_size"
        ],
        disable_tqdm=False,
        predict_with_generate=True,
        generation_num_beams=5,
        generation_max_length=200,
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
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    logger.info("📝 Running evaluation...")
    results = trainer.evaluate()

    if experiment_config.use_mlflow:
        import mlflow

        mlflow.log_params(config_dict)
        mlflow.log_param("experiment_type", "baseline_evaluation")
        mlflow.log_param("dataset_subset", experiment_config.dataset_subset)
        mlflow.log_param("training_enabled", False)
        mlflow.log_param("generation_num_beams", 5)
        mlflow.log_param("submission_style_evaluation", True)

        for key, value in results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
                # Log final metrics with special prefix for easy filtering
                if key.startswith("final_"):
                    mlflow.log_metric(f"eval_{key}", value)

    return results


def run_training(config_dict: dict, experiment_config: ExperimentConfig):
    """Run the full training process."""
    logger.info(
        f"🚀 Starting training with dataset subset: {experiment_config.dataset_subset}"
    )

    logger.info("📊 Creating datasets...")
    train_ds = salt.dataset.create(config_dict["train"], verbose=True)
    valid_ds = salt.dataset.create(config_dict["validation"])

    logger.info("📚 Loading prompts dataset...")
    try:
        import datasets

        ds = datasets.load_dataset(
            "jq/kinyarwanda-speech-hackathon", split="train", num_proc=4
        )
        text = list(ds["text"])
        prompts = list(ds["prompt"])
        sentence_to_prompt = {t: p for t, p in zip(text, prompts)}
        logger.info(f"✅ Loaded {len(sentence_to_prompt)} prompts")
    except Exception as e:
        logger.warning(f"⚠️ Could not load prompts: {e}. Continuing without prompts.")
        sentence_to_prompt = {}

    logger.info("🤖 Loading model and processor...")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        config_dict["pretrained_model"]
    )
    processor = transformers.WhisperProcessor.from_pretrained(
        config_dict["pretrained_model"], language=None, task="transcribe"
    )
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        config_dict["pretrained_model"]
    )

    # Configure model
    model.config.suppress_tokens = []
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    logger.info("🔧 Preparing datasets...")
    train_data = train_ds.map(
        lambda x: prepare_dataset(x, sentence_to_prompt, feature_extractor, processor),
        remove_columns=["source", "target"],
    )
    val_data = valid_ds.map(
        lambda x: prepare_dataset(x, sentence_to_prompt, feature_extractor, processor),
        remove_columns=["source", "target"],
    )

    compute_metrics = create_evaluation_compute_metrics_fn(valid_ds, processor)

    # Setup PEFT if enabled
    if config_dict.get("use_peft", False):
        logger.info("🔧 Setting up PEFT...")
        model = peft.prepare_model_for_kbit_training(model)
        lora_config = peft.LoraConfig(**config_dict["lora_config"])
        model.enable_input_require_grads()
        model = peft.get_peft_model(model, lora_config)
        model.config.use_cache = False
        model.print_trainable_parameters()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id
    )

    training_args = transformers.Seq2SeqTrainingArguments(
        **config_dict["training_args"],
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

    # Log to MLflow
    if experiment_config.use_mlflow:
        import mlflow

        mlflow.log_params(config_dict)
        mlflow.log_param("experiment_type", "fine_tuning")
        mlflow.log_param("dataset_subset", experiment_config.dataset_subset)
        mlflow.log_param("training_enabled", True)
        mlflow.log_param("generation_num_beams", 5)
        mlflow.log_param("submission_style_evaluation", True)

        for key, value in results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
                # Log final metrics with special prefix for easy filtering
                if key.startswith("final_"):
                    mlflow.log_metric(f"eval_{key}", value)

    # Push to hub if enabled
    if config_dict["training_args"].get("push_to_hub", False):
        logger.info("📤 Pushing to Hub...")
        processor.push_to_hub(
            config_dict["training_args"]["hub_model_id"], private=True
        )
        model.push_to_hub(config_dict["training_args"]["hub_model_id"], private=True)

    return results


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return ExperimentConfig(**config_dict)
    except Exception as e:
        logger.error(f"❌ Failed to load config from {config_path}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on Kinyarwanda speech data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file",
    )

    args = parser.parse_args()

    # Load configuration
    experiment_config = load_experiment_config(args.config)

    # Set seeds for reproducibility
    torch.manual_seed(experiment_config.seed)
    np.random.seed(experiment_config.seed)

    logger.info("=" * 60)
    logger.info(f"🎯 Starting experiment: {experiment_config.experiment_name}")
    logger.info(f"📋 Configuration: {args.config}")
    logger.info(f"🎲 Seed: {experiment_config.seed}")
    logger.info("=" * 60)

    # Setup logging backends
    setup_logging_backends(
        experiment_config.use_wandb,
        experiment_config.use_mlflow,
        experiment_config.experiment_name,
    )

    # Generate training configuration
    config_dict = get_training_config(experiment_config)

    # Add timestamp to experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_config.experiment_name = (
        f"{experiment_config.experiment_name}_{timestamp}"
    )
    config_dict["training_args"]["output_dir"] = experiment_config.experiment_name
    config_dict["training_args"]["hub_model_id"] = (
        f"jq/{experiment_config.experiment_name}"
    )

    # Create output directory
    os.makedirs(experiment_config.experiment_name, exist_ok=True)

    # Save configurations
    with open(
        os.path.join(experiment_config.experiment_name, "experiment_config.yaml"), "w"
    ) as f:
        yaml.dump(experiment_config.__dict__, f, default_flow_style=False)

    with open(
        os.path.join(experiment_config.experiment_name, "training_config.yaml"), "w"
    ) as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Run experiment
    try:
        if experiment_config.run_training:
            results = run_training(config_dict, experiment_config)
        else:
            results = run_baseline_evaluation(config_dict, experiment_config)

        # Display and save results
        logger.info("=" * 60)
        logger.info("🎉 FINAL RESULTS:")
        logger.info("📊 EVALUATION METRICS:")
        if "final_WER" in results:
            logger.info(f"   🎯 Final WER: {results['final_WER']:.4f}")
            logger.info(f"   🎯 Final CER: {results['final_CER']:.4f}")
            logger.info(f"   🎯 Final Score: {results['final_score']:.4f}")
        logger.info("📊 SALT METRICS (training monitoring):")
        for key, value in results.items():
            if not key.startswith("final_"):
                logger.info(f"   {key}: {value}")
        logger.info("=" * 60)

        results_path = os.path.join(experiment_config.experiment_name, "results.yaml")
        with open(results_path, "w") as f:
            yaml.dump(results, f)

        logger.info(f"💾 Results saved to: {results_path}")
        logger.info(f"✅ Experiment completed: {experiment_config.experiment_name}")

    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        raise

    finally:
        # Clean up MLflow
        if experiment_config.use_mlflow:
            try:
                import mlflow

                mlflow.end_run()
            except:
                pass


if __name__ == "__main__":
    main()

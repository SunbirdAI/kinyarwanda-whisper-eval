#!/usr/bin/env python3
"""
Swahili Whisper fine-tuning script.
Usage:
  uv run python train.py --config configs/train_32h.yaml
  uv run python train.py --config configs/train_1h.yaml
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
from datasets import load_dataset, Dataset as HFDataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from datetime import datetime
import logging
from transformers import EarlyStoppingCallback
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------
# Config & Collator
# ----------------------------
@dataclass
class ExperimentConfig:
    experiment_name: str
    dataset_subset: str = "audio_50h"
    train_duration_hours: Optional[float] = None  # None => use full split
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
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        # whisper wants float16/bfloat16 features
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # remove BOS if present at position 0
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ----------------------------
# Logging backends
# ----------------------------
def setup_logging_backends(use_wandb: bool, use_mlflow: bool, experiment_name: str):
    """Setup logging backends."""
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

            os.environ["MLFLOW_EXPERIMENT_NAME"] = "whisper-kikuyu-eval"
            mlflow.set_tracking_uri(
                "https://mlflow-sunbird-ce0ecfc14244.herokuapp.com/"
            )
            mlflow.system_metrics.enable_system_metrics_logging()
            mlflow.start_run(run_name=experiment_name)
            logger.info("✅ MLflow initialized")
        except ImportError:
            logger.error("❌ mlflow not installed")
            raise


# ----------------------------
# Preprocessing
# ----------------------------
def prepare_dataset(
    example, sentence_to_prompt, feature_extractor, processor, p_prompt: float = 0.0
):
    """Map-style preprocessing for SALT-created items."""
    audio = example["source"]  # PCM at 16k
    input_features = feature_extractor(
        audio,
        sampling_rate=16000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        do_normalize=True,
    ).input_features[0]

    # Encode target text to label ids
    labels = processor.tokenizer(str(example["target"])).input_ids

    # Insert language ID token at position 1
    labels.insert(
        1, salt.constants.SALT_LANGUAGE_TOKENS_WHISPER[example["target.language"]]
    )

    # Optional prompt (kept deterministic if desired by swapping RNG below)
    prompt = sentence_to_prompt.get(example["target"], None)
    if prompt and np.random.random() < p_prompt:
        prompt_ids = list(processor.get_prompt_ids(prompt))
        labels = prompt_ids + labels

    labels = labels[:448]

    return {
        "input_features": input_features,
        "labels": np.array(labels),
        "source.language": example["source.language"],
        "target.language": example["target.language"],
    }


def build_duration_subset(
    base_iterable_ds,
    *,
    limit_hours: float,
    seed: int,
    sentence_to_prompt: dict,
    feature_extractor,
    processor,
    target_sr: int = 16000,
    shuffle_buffer: int = 100_000,
    p_prompt: float = 0.0,
) -> HFDataset:
    """
    Deterministically materialize a finite N-hour slice from a streaming SALT dataset.
    Returns a map-style HF Dataset so Trainer can do multiple epochs w/o re-opening shards.
    """
    # Buffered deterministic shuffle on the iterable
    shuffled = base_iterable_ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    limit_seconds = int(limit_hours * 3600)
    current_seconds = 0

    # RNG for prompt decision; swap to hash(text) if you want per-sample determinism
    rng = np.random.RandomState(seed)

    def _gen():
        nonlocal current_seconds
        pbar = tqdm(total=limit_seconds, unit="sec", desc="Building subset")

        for ex in shuffled:
            # duration by sample length / sr
            dur = len(ex["source"]) / float(target_sr)
            if current_seconds > 0 and current_seconds + dur > limit_seconds:
                break

            audio = ex["source"]
            input_features = feature_extractor(
                audio,
                sampling_rate=target_sr,
                device="cuda" if torch.cuda.is_available() else "cpu",
                do_normalize=True,
            ).input_features[0]

            labels = processor.tokenizer(str(ex["target"])).input_ids
            labels.insert(
                1, salt.constants.SALT_LANGUAGE_TOKENS_WHISPER[ex["target.language"]]
            )

            prompt = sentence_to_prompt.get(ex["target"], None)
            if prompt and rng.rand() < p_prompt:
                prompt_ids = list(processor.get_prompt_ids(prompt))
                labels = prompt_ids + labels

            labels = labels[:448]

            current_seconds += dur
            pbar.update(int(dur))

            # Return lists (not np arrays) for max compatibility
            yield {
                "input_features": input_features,  # list[float]
                "labels": labels,  # list[int]
                "source.language": ex["source.language"],
                "target.language": ex["target.language"],
            }
        pbar.close()

    ds = HFDataset.from_generator(_gen)
    ds = ds.with_format("torch")
    return ds


# ----------------------------
# SALT config
# ----------------------------
def build_salt_config(experiment_config: ExperimentConfig) -> dict:
    """Build the SALT config (kept close to your original)."""
    config_yaml = f"""
pretrained_model: openai/whisper-large-v3
num_workers: 12
use_peft: false

training_args:
    output_dir: {experiment_config.experiment_name}
    per_device_train_batch_size: 8
    per_device_eval_batch_size: 8
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
        - path: evie-8/kikuyu-data
          name: {experiment_config.dataset_subset}
          split: train
          num_proc: 10
        - path: evie-8/kikuyu-data
          name: {experiment_config.dataset_subset}
          split: train[:100]
          num_proc: 10
    source:
      type: speech
      language: [kik]
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
      language: [kik]
    shuffle: True

validation:
    huggingface_load:
        - path: evie-8/kikuyu-data
          name: {experiment_config.dataset_subset}
          split: dev_test
    source:
      type: speech
      language: [kik]
      preprocessing:
        - set_sample_rate:
            rate: 16_000
    target:
      type: text
      language: [kik]
      preprocessing:
        - lower_case
        - clean_and_remove_punctuation:
            allowed_punctuation: "'"
"""
    return yaml.safe_load(config_yaml)


# ----------------------------
# Entry
# ----------------------------
def load_experiment_config(config_path: str) -> ExperimentConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return ExperimentConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(description="Train Whisper on Swahili speech data")
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
    if experiment_config.train_duration_hours is not None:
        logger.info(
            f"🕒 Train duration target: {experiment_config.train_duration_hours}h"
        )
    logger.info("=" * 60)

    # Setup logging
    setup_logging_backends(
        experiment_config.use_wandb,
        experiment_config.use_mlflow,
        experiment_config.experiment_name,
    )

    # Build SALT config
    config = build_salt_config(experiment_config)

    # Add timestamp to output directory + hub id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{experiment_config.experiment_name}_{timestamp}"
    config["training_args"]["output_dir"] = output_dir
    config["training_args"][
        "hub_model_id"
    ] = f"akera/{experiment_config.experiment_name}_{timestamp}"

    # Persist configs alongside run
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "experiment_config.yaml"), "w") as f:
        yaml.dump(experiment_config.__dict__, f, default_flow_style=False)
    with open(os.path.join(output_dir, "full_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info("📊 Creating datasets using SALT...")
    train_ds_raw = salt.dataset.create(config["train"], verbose=True)
    valid_ds_raw = salt.dataset.create(config["validation"])

    logger.info("📚 Loading prompts dataset...")
    try:
        ds_prompts = load_dataset(
            "evie-8/kikuyu-data",
            name=experiment_config.dataset_subset,
            split="train",
        )
        sentence_to_prompt = {
            t: p for t, p in zip(ds_prompts["text"], ds_prompts["prompt"])
        }
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
        config["pretrained_model"],
        torch_dtype=torch.bfloat16,
    )

    logger.info("🔧 Preparing datasets...")

    # ---- Duration-limited vs full ----
    if experiment_config.train_duration_hours is not None:
        logger.info(
            f"🎯 Building ~{experiment_config.train_duration_hours}h deterministic subset"
        )
        train_data = build_duration_subset(
            base_iterable_ds=train_ds_raw,
            limit_hours=experiment_config.train_duration_hours,
            seed=experiment_config.seed,
            sentence_to_prompt=sentence_to_prompt,
            feature_extractor=feature_extractor,
            processor=processor,
            target_sr=16000,
            shuffle_buffer=100_000,
            p_prompt=0.0,
        )
        val_data = valid_ds_raw.map(
            lambda x: prepare_dataset(
                x, sentence_to_prompt, feature_extractor, processor
            ),
            remove_columns=["source", "target"],
        )
        # NOTE: Because train_data is map-style now, multiple epochs are fine.
        # Keep max_steps + early stopping if you want; no slow shard reloads.
    else:
        # Full split as you had before
        train_data = train_ds_raw.map(
            lambda x: prepare_dataset(
                x, sentence_to_prompt, feature_extractor, processor
            ),
            remove_columns=["source", "target"],
        )
        val_data = valid_ds_raw.map(
            lambda x: prepare_dataset(
                x, sentence_to_prompt, feature_extractor, processor
            ),
            remove_columns=["source", "target"],
        )

    # Compute metrics
    compute_metrics = salt.metrics.multilingual_eval_fn(
        valid_ds_raw,
        [evaluate.load("wer"), evaluate.load("cer")],
        processor.tokenizer,
        log_first_N_predictions=3,
        speech_processor=processor,
    )

    # Model config
    model.config.suppress_tokens = []
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id
    )

    # Training args
    training_args = transformers.Seq2SeqTrainingArguments(
        **config["training_args"],
        disable_tqdm=False,
        remove_unused_columns=False,  # important for custom/streaming shapes
        report_to=[
            platform
            for platform, use in [
                ("wandb", experiment_config.use_wandb),
                ("mlflow", experiment_config.use_mlflow),
            ]
            if use
        ],
    )

    # Trainer
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

    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    else:
        gpu_name = "cpu"
        vram_gb = 0.0

    logger.info("🏃 Starting training...")
    trainer.train()

    logger.info("📝 Running final evaluation...")
    results = trainer.evaluate()

    # Log to MLflow if enabled
    if experiment_config.use_mlflow:
        import mlflow

        # Log flattened training args/config selectively (avoid huge nested dump)
        mlflow.log_param("experiment_name", experiment_config.experiment_name)
        mlflow.log_param("dataset_subset", experiment_config.dataset_subset)
        if experiment_config.train_duration_hours is not None:
            mlflow.log_param(
                "train_duration_hours", experiment_config.train_duration_hours
            )
        mlflow.log_param("gpu_name", gpu_name)
        mlflow.log_param("vram_gb", vram_gb)

        for k, v in results.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        mlflow.end_run()

    # Save results locally
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

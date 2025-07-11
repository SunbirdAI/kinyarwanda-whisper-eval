#!/usr/bin/env python3
"""
Training script for Kinyarwanda Whisper fine-tuning.

Usage:
  python train.py --config configs/train_50h.yaml
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
from datasets import load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datetime import datetime
import logging

from utils import prepare_dataset_hf_with_augmentations, prepare_dataset_salt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    experiment_name: str
    dataset_subset: str = "audio_50h"
    use_wandb: bool = False
    use_mlflow: bool = True
    push_to_hub: bool = False
    hub_model_id: str = None
    seed: int = 42


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def setup_logging_backends(use_wandb: bool, use_mlflow: bool, run_name: str):
    if use_wandb:
        try:
            import wandb

            os.environ["WANDB_LOG_MODEL"] = "True"
            os.environ["WANDB_WATCH"] = "all"
            wandb.login()
            logger.info("✅ Weights & Biases initialized")
        except ImportError:
            logger.error("❌ wandb not installed. pip install wandb")
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
            mlflow.start_run(run_name=run_name)
            logger.info("✅ MLflow initialized")
        except ImportError:
            logger.error("❌ mlflow not installed. pip install mlflow psutil pynvml")
            raise


def get_training_config(exp_cfg: ExperimentConfig) -> dict:
    hub_model_id = exp_cfg.hub_model_id or f"akera/{exp_cfg.experiment_name}"
    tpl = f"""
pretrained_model: openai/whisper-large-v3
num_workers: 4
use_peft: False

training_args:
    per_device_train_batch_size: 8
    per_device_eval_batch_size: 8
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
    save_steps: 1000
    eval_steps: 200
    logging_steps: 200
    load_best_model_at_end: True
    metric_for_best_model: loss
    greater_is_better: False
    push_to_hub: True
    hub_model_id: {hub_model_id}
    save_total_limit: 2

validation:
    huggingface_load:
      - path: jq/kinyarwanda-speech-hackathon
        split: dev_test[:300]
    source:
      type: speech
      language: [kin]
      preprocessing:
        - set_sample_rate:
            rate: 16000
    target:
      type: text
      language: [kin]
      preprocessing:
        - lower_case
        - clean_and_remove_punctuation:
            allowed_punctuation: "'"
"""
    return yaml.safe_load(tpl)


def load_experiment_config(path: str) -> ExperimentConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    exp_cfg = load_experiment_config(args.config)
    torch.manual_seed(exp_cfg.seed)
    np.random.seed(exp_cfg.seed)

    logger.info("=" * 60)
    logger.info(f"🎯 Experiment: {exp_cfg.experiment_name}")
    logger.info(f"📋 Config file: {args.config}")
    logger.info(f"🎲 Seed: {exp_cfg.seed}")
    logger.info("=" * 60)

    setup_logging_backends(
        exp_cfg.use_wandb, exp_cfg.use_mlflow, exp_cfg.experiment_name
    )

    cfg = get_training_config(exp_cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{exp_cfg.experiment_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "experiment_config.yaml"), "w") as f:
        yaml.safe_dump(vars(exp_cfg), f)
    with open(os.path.join(output_dir, "training_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    try:
        ds_prompts = load_dataset(
            "evie-8/kinyarwanda-speech-hackathon",
            name=exp_cfg.dataset_subset,
            split="train",
        )
        sentence_to_prompt = dict(zip(ds_prompts["text"], ds_prompts["prompt"]))
        logger.info(f"✅ Loaded {len(sentence_to_prompt)} prompts")
    except Exception as e:
        logger.warning(f"⚠️ Could not load prompts ({e}); continuing.")
        sentence_to_prompt = {}

    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        cfg["pretrained_model"]
    )
    processor = transformers.WhisperProcessor.from_pretrained(
        cfg["pretrained_model"], language=None, task="transcribe"
    )
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        cfg["pretrained_model"]
    )
    model.config.use_cache = False

    valid_ds = salt.dataset.create(cfg["validation"], verbose=True)
    val_data = valid_ds.map(
        lambda ex: prepare_dataset_salt(
            ex, sentence_to_prompt, feature_extractor, processor
        ),
        remove_columns=["source", "target"],
    )
    compute_metrics = salt.metrics.multilingual_eval_fn(
        valid_ds,
        [evaluate.load("wer"), evaluate.load("cer")],
        processor.tokenizer,
        log_first_N_predictions=3,
        speech_processor=processor,
    )

    raw_train = load_dataset(
        "evie-8/kinyarwanda-speech-hackathon",
        name=exp_cfg.dataset_subset,
        split="train",
        streaming=True,
    )
    train_data = raw_train.map(
        lambda ex: prepare_dataset_hf_with_augmentations(
            ex, sentence_to_prompt, feature_extractor, processor
        ),
        remove_columns=raw_train.column_names,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=output_dir,
        **cfg["training_args"],
        disable_tqdm=False,
        report_to=[
            p
            for p, use in [("wandb", exp_cfg.use_wandb), ("mlflow", exp_cfg.use_mlflow)]
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

    if exp_cfg.use_mlflow:
        import mlflow

        mlflow.log_params(cfg)
        for k, v in results.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        mlflow.end_run()

    if exp_cfg.push_to_hub:
        processor.push_to_hub(exp_cfg.hub_model_id, private=True)
        model.push_to_hub(exp_cfg.hub_model_id, private=True)

    with open(os.path.join(output_dir, "results.yaml"), "w") as f:
        yaml.safe_dump(results, f)

    logger.info("🎉 TRAINING COMPLETED!")
    logger.info(f"💾 Results in {output_dir}/results.yaml")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluation script for Kinyarwanda Whisper models.
Uses direct model.generate for fast, batched GPU throughput
and shows a live tqdm progress bar over 300 samples.
"""

import argparse
import os
import math
import itertools
from getpass import getpass
import logging

import torch
import datasets
import jiwer
import string
import mlflow
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import salt.constants


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def strip_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def normalise(texts):
    return [strip_punctuation(t.lower()) for t in texts]


def setup_mlflow(model_path: str):
    if "MLFLOW_TRACKING_USERNAME" not in os.environ:
        os.environ["MLFLOW_TRACKING_USERNAME"] = getpass("Enter MLFLOW_TRACKING_USERNAME: ")
    if "MLFLOW_TRACKING_PASSWORD" not in os.environ:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass("Enter MLFLOW_TRACKING_PASSWORD: ")

    os.environ["MLFLOW_EXPERIMENT_NAME"] = "whisper-kinyarwanda-eval"
    mlflow.set_tracking_uri("https://mlflow-sunbird-ce0ecfc14244.herokuapp.com/")

    run_name = "eval_" + os.path.basename(model_path)
    mlflow.start_run(run_name=run_name)
    logger.info(f"✅ MLflow initialized with run: {run_name}")


def load_validation_dataset():
    logger.info("📊 Loading validation dataset..")
    ds = datasets.load_dataset(
        "jq/kinyarwanda-speech-hackathon",
        split="dev_test[:300]"
    )
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    logger.info(f"✅ Loaded {len(ds)} samples")
    return ds


def evaluate_model(model_path: str, batch_size: int):
    # Load dataset & snapshot labels
    ds = load_validation_dataset()
    total = len(ds)
    labels = [ex["text"] for ex in ds]

    # Build an audio generator from the IterableDataset
    audio_iter = (ex["audio"]["array"] for ex in ds.to_iterable_dataset())

    # Load processor + model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Tell the processor not to inject any language/task tokens
    processor = WhisperProcessor.from_pretrained(
        model_path, language=None, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch_dtype).to(device)
    model.eval()


    # **Disable any forced-decoder tokens** that might be baked into config
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    logger.info(f"✅ Model & processor loaded on {device} @ {torch_dtype}")

    # Batch + generate with tqdm
    num_batches = math.ceil(total / batch_size)
    logger.info(f"🔄 Running inference in {num_batches} batches (size={batch_size})...")
    predictions = []

    gen_kwargs = {
        "num_beams": 5,
        "max_length": 200,
    }

    for _ in tqdm(range(num_batches), desc="Transcribing"):
        batch_audio = list(itertools.islice(audio_iter, batch_size))
        if not batch_audio:
            break
        feat = processor.feature_extractor(
            batch_audio,
            sampling_rate=16_000,
            return_tensors="pt",
            return_attention_mask=True
        )

        inputs = feat.input_features.to(device, dtype=torch_dtype)
        attention_mask = feat.attention_mask.to(device)
        
        gen_ids = model.generate(inputs, attention_mask=attention_mask, **gen_kwargs)

        texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
        predictions.extend(texts)

    logger.info("✅ Inference completed")

    # Compute metrics
    logger.info("📊 Calculating WER/CER...")
    norm_preds = normalise(predictions)
    norm_labels = normalise(labels)
    wer = jiwer.wer(norm_labels, norm_preds)
    cer = jiwer.cer(norm_labels, norm_preds)
    score = 1.0 - (0.6 * cer + 0.4 * wer)

    return {
        "final_WER": wer,
        "final_CER": cer,
        "final_score": score,
        "predictions": predictions,
        "labels": labels,
    }


def log_results(results: dict, model_path: str, batch_size: int):
    logger.info("📝 Logging results to MLflow...")
    mlflow.log_param("model_path", model_path)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("dataset_subset", "dev_test[:300]")
    mlflow.log_param("num_samples", len(results["labels"]))

    mlflow.log_metric("final_WER", results["final_WER"])
    mlflow.log_metric("final_CER", results["final_CER"])
    mlflow.log_metric("final_score", results["final_score"])
    logger.info("✅ MLflow logging done")


def main():
    parser = argparse.ArgumentParser(
        description="Fast eval of Kinyarwanda Whisper w/ tqdm bar",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", required=True, help="HF ID or local path")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--no_mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🎯 KINYARWANDA WHISPER EVALUATION")
    logger.info(f"Model:      {args.model_path}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 60)

    try:
        if not args.no_mlflow:
            setup_mlflow(args.model_path)

        results = evaluate_model(args.model_path, args.batch_size)

        logger.info("=" * 60)
        logger.info("🎉 EVALUATION RESULTS")
        logger.info(f"  Final WER:   {results['final_WER']:.4f}")
        logger.info(f"  Final CER:   {results['final_CER']:.4f}")
        logger.info(f"  Final Score: {results['final_score']:.4f}")
        logger.info("📝 Sample predictions:")
        for i in range(min(3, len(results["predictions"]))):
            logger.info(f"    Label: {results['labels'][i]!r}")
            logger.info(f"    Pred:  {results['predictions'][i]!r}")
            logger.info("    ---")

        if not args.no_mlflow:
            log_results(results, args.model_path, args.batch_size)

        logger.info("✅ Done!")
    except Exception:
        logger.error("❌ Evaluation failed", exc_info=True)
    finally:
        if not args.no_mlflow and mlflow.active_run():
            mlflow.end_run()


if __name__ == "__main__":
    main()


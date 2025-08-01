#!/usr/bin/env python3
"""
Data utilities for creating hour-based splits and saving them as DatasetDict.
Creates actual dataset splits that behave exactly like Kinyarwanda pre-existing subsets.
"""

import logging
import numpy as np
import os
from datasets import load_dataset, Dataset, DatasetDict
from typing import Tuple, Dict
import librosa
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


def get_audio_duration(audio_array: np.ndarray, sampling_rate: int = 16000) -> float:
    """Get duration of audio in hours"""
    return len(audio_array) / sampling_rate / 3600


def get_split_cache_path(
    dataset_name: str,
    language_subset: str,
    target_hours: float,
    test_size: int,
    seed: int,
) -> Path:
    """Generate cache path for split based on parameters"""
    # Create a hash of the parameters to ensure unique cache dirs
    params_str = (
        f"{dataset_name}_{language_subset}_{target_hours}h_{test_size}test_seed{seed}"
    )
    cache_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

    cache_dir = Path(
        f".cache/dataset_splits/{language_subset}_{target_hours}h_{cache_hash}"
    )
    return cache_dir


def create_and_save_hour_based_splits(
    dataset_name: str,
    language_subset: str,
    target_hours: float,
    test_size: int = 300,
    seed: int = 42,
    force_recreate: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Create deterministic hour-based splits and save them as HF DatasetDict.
    Returns info for loading with SALT's huggingface_load.

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'evie-8/afrivoices')
        language_subset: Language subset (e.g., 'sna', 'ful', 'lin')
        target_hours: Target hours for training split
        test_size: Number of samples to hold out for testing
        seed: Random seed for deterministic splits
        force_recreate: Force recreation even if cache exists

    Returns:
        train_split: Dictionary with train dataset info
        test_split: Dictionary with test dataset info
    """
    logger.info(f"Creating {target_hours}h split for {language_subset}")

    # Check cache first
    cache_dir = get_split_cache_path(
        dataset_name, language_subset, target_hours, test_size, seed
    )

    if (
        not force_recreate
        and cache_dir.exists()
        and (cache_dir / "dataset_dict.json").exists()
    ):
        logger.info(f"✅ Found cached splits at {cache_dir}")

        # Load cached info
        info_file = cache_dir / "split_info.txt"
        if info_file.exists():
            with open(info_file, "r") as f:
                lines = f.readlines()
                train_samples = int(lines[0].split(":")[1].strip())
                actual_hours = float(lines[1].split(":")[1].strip())
                test_samples = int(lines[2].split(":")[1].strip())
        else:
            # Fallback - approximate values
            train_samples = 0
            test_samples = test_size
            actual_hours = target_hours

        return {
            "dataset_path": str(cache_dir),
            "actual_hours": actual_hours,
            "num_samples": train_samples,
            "is_cached": True,
        }, {
            "dataset_path": str(cache_dir),
            "num_samples": test_samples,
            "is_cached": True,
        }

    # Create splits from scratch
    logger.info("Creating splits from scratch...")

    # Load the full train split
    ds = load_dataset(dataset_name, name=language_subset, split="train")

    # Set seed for deterministic behavior
    np.random.seed(seed)
    indices = np.arange(len(ds))
    np.random.shuffle(indices)

    # Reserve test samples first
    test_indices = indices[:test_size]
    remaining_indices = indices[test_size:]

    # Calculate cumulative hours for remaining samples
    logger.info("Calculating audio durations for training split...")
    cumulative_hours = 0
    train_indices = []

    for idx in remaining_indices:
        sample = ds[int(idx)]
        # Load audio to get duration
        audio_array = sample["audio"]["array"]
        duration_hours = get_audio_duration(
            audio_array, sample["audio"]["sampling_rate"]
        )

        if cumulative_hours + duration_hours <= target_hours:
            train_indices.append(int(idx))
            cumulative_hours += duration_hours
        else:
            break

    logger.info(
        f"Selected {len(train_indices)} samples for {cumulative_hours:.2f}h of training data"
    )
    logger.info(f"Reserved {len(test_indices)} samples for testing")

    # Create the actual dataset splits
    test_dataset = ds.select(test_indices)

    if len(train_indices) > 0:
        train_dataset = ds.select(train_indices)
        # Create DatasetDict with both splits
        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    else:
        # For baseline (no training), only create test split to avoid empty dataset error
        logger.info("Creating DatasetDict with only test split (no training data)")
        dataset_dict = DatasetDict({"test": test_dataset})

    # Save the DatasetDict to cache
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"💾 Saving DatasetDict ({len(train_indices)} train, {len(test_dataset)} test) to {cache_dir}"
    )
    dataset_dict.save_to_disk(str(cache_dir))

    # Save split info for future reference
    info_file = cache_dir / "split_info.txt"
    with open(info_file, "w") as f:
        f.write(f"train_samples: {len(train_indices)}\n")
        f.write(f"actual_hours: {cumulative_hours}\n")
        f.write(f"test_samples: {len(test_dataset)}\n")
        f.write(f"target_hours: {target_hours}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"dataset_name: {dataset_name}\n")
        f.write(f"language_subset: {language_subset}\n")

    logger.info(f"✅ DatasetDict saved to {cache_dir}")

    return {
        "dataset_path": str(cache_dir),
        "actual_hours": cumulative_hours,
        "num_samples": len(train_indices),
        "is_cached": False,
    }, {
        "dataset_path": str(cache_dir),
        "num_samples": len(test_dataset),
        "is_cached": False,
    }


def get_language_code_mapping():
    """Get mapping of dataset subsets to language codes"""
    return {
        "sna": "sna",  # Shona
        "ful": "ful",  # Fulani
        "lin": "lin",  # Lingala
        "kin": "kin",  # Kinyarwanda (for backward compatibility)
    }


def cleanup_old_cache(keep_recent: int = 5):
    """Clean up old cached splits, keeping only the most recent ones"""
    cache_base = Path(".cache/dataset_splits")
    if not cache_base.exists():
        return

    # Group by language
    language_dirs = {}
    for cache_dir in cache_base.iterdir():
        if cache_dir.is_dir():
            lang = cache_dir.name.split("_")[0]
            if lang not in language_dirs:
                language_dirs[lang] = []
            language_dirs[lang].append(cache_dir)

    # Keep only recent ones per language
    for lang, dirs in language_dirs.items():
        if len(dirs) > keep_recent:
            # Sort by modification time
            dirs.sort(key=lambda x: x.stat().st_mtime)
            old_dirs = dirs[:-keep_recent]

            for old_dir in old_dirs:
                logger.info(f"🗑️ Cleaning up old cache: {old_dir}")
                import shutil

                shutil.rmtree(old_dir, ignore_errors=True)

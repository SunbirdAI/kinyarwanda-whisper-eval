#!/usr/bin/env python3
"""
Data utilities for creating hour-based splits deterministically
"""

import logging
import numpy as np
from datasets import load_dataset
from typing import Tuple, Dict
import librosa

logger = logging.getLogger(__name__)


def get_audio_duration(audio_array: np.ndarray, sampling_rate: int = 16000) -> float:
    """Get duration of audio in hours"""
    return len(audio_array) / sampling_rate / 3600


def create_hour_based_splits(
    dataset_name: str,
    language_subset: str,
    target_hours: float,
    test_size: int = 300,
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    """
    Create deterministic hour-based splits from the train split.

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'evie-8/afrivoices')
        language_subset: Language subset (e.g., 'sna', 'ful', 'lin')
        target_hours: Target hours for training split
        test_size: Number of samples to hold out for testing
        seed: Random seed for deterministic splits

    Returns:
        train_split: Dictionary with train data info
        test_split: Dictionary with test data info
    """
    logger.info(f"Creating {target_hours}h split for {language_subset}")

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
    logger.info("Calculating audio durations...")
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

    train_split = {
        "dataset_name": dataset_name,
        "subset": language_subset,
        "split": (
            f"train[{','.join(map(str, train_indices))}]"
            if train_indices
            else "train[:0]"
        ),
        "indices": train_indices,
        "actual_hours": cumulative_hours,
    }

    test_split = {
        "dataset_name": dataset_name,
        "subset": language_subset,
        "split": f"train[{','.join(map(str, test_indices))}]",
        "indices": test_indices,
        "size": len(test_indices),
    }

    return train_split, test_split


def get_language_code_mapping():
    """Get mapping of dataset subsets to language codes"""
    return {
        "sna": "sna",  # Shona
        "ful": "ful",  # Fulani
        "lin": "lin",  # Lingala
        "kin": "kin",  # Kinyarwanda (for backward compatibility)
    }

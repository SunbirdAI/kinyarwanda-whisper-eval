"""
SALT Preprocessing Utilities
import salt.preprocessing
"""

import librosa
import numpy as np
import re
import string
import cleantext
import salt.constants


def clean_and_remove_punctuation(text, allowed_punctuation="'"):
    # Use cleantext.clean first
    text = cleantext.clean(text, to_ascii=False, no_punct=False)

    # Remove punctuation except allowed
    punct = list(string.punctuation)
    if allowed_punctuation:
        for allowed in allowed_punctuation:
            if allowed in punct:
                punct.remove(allowed)

    return "".join([c for c in text if c not in punct])


def augment_audio_speed(audio, p=0.2, low=0.95, high=1.15):
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)
    if not len(audio):
        return audio

    if np.random.random() < p:
        speed_factor = np.random.uniform(low, high)
        audio = librosa.effects.time_stretch(audio, rate=speed_factor)
    return audio


def augment_audio_noise(
    audio, max_relative_amplitude=0.3, min_coverage=0.4, max_coverage=1.0
):
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)
    if not len(audio):
        return audio

    # Use 99th percentile as reference amplitude
    x_reference_amplitude = np.percentile(np.abs(audio), 99)
    amplitude = np.random.uniform(0, max_relative_amplitude) * x_reference_amplitude

    # Apply noise to random segment
    coverage = np.random.uniform(min_coverage, max_coverage)
    num_samples_to_affect = int(len(audio) * coverage)
    start_index = np.random.randint(0, len(audio) - num_samples_to_affect)

    # Generate white noise for the segment
    noise = np.random.uniform(-amplitude, amplitude, size=num_samples_to_affect)

    # Apply noise to chosen segment
    audio_with_noise = np.copy(audio)
    audio_with_noise[start_index : start_index + num_samples_to_affect] += noise

    return audio_with_noise


def set_sample_rate(audio, current_rate, target_rate, p=1.0):
    if current_rate != target_rate:
        if p == 1.0 or np.random.random() < p:
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            audio = librosa.resample(audio, orig_sr=current_rate, target_sr=target_rate)
    return audio


def normalize_audio(audio):
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)
    # SALT's exact normalization: subtract mean, then normalize by max + epsilon
    audio = audio - np.mean(audio)
    audio = audio / (np.max(np.abs(audio)) + 1e-3)
    return audio


def prepare_dataset_hf_with_augmentations(
    example, sentence_to_prompt, feature_extractor, processor, p_prompt=0.0
):
    if "audio" in example:
        audio_array = example["audio"]["array"]
    elif "source" in example:
        audio_array = example["source"]
    else:
        raise KeyError("No audio field in example")

    if "target" in example:
        text = example["target"]
    elif "text" in example:
        text = example["text"]
    else:
        raise KeyError("No text field in example")

    sample_rate = 16000

    audio_array = set_sample_rate(audio_array, sample_rate, 8000, p=0.05)
    audio_array = set_sample_rate(audio_array, 8000, sample_rate, p=1.0)
    audio_array = normalize_audio(audio_array)
    audio_array = augment_audio_speed(audio_array, p=0.2, low=0.95, high=1.15)
    audio_array = augment_audio_noise(audio_array, max_relative_amplitude=0.3)

    input_features = feature_extractor(
        audio_array, sampling_rate=sample_rate, do_normalize=True
    ).input_features[0]

    text = text.lower()
    text = clean_and_remove_punctuation(text, allowed_punctuation="'")

    labels = processor.tokenizer(str(text)).input_ids
    labels.insert(1, salt.constants.SALT_LANGUAGE_TOKENS_WHISPER["kin"])

    prompt = sentence_to_prompt.get(text, None)
    if prompt and np.random.random() < p_prompt:
        labels = list(processor.get_prompt_ids(prompt)) + labels

    return {
        "input_features": input_features,
        "labels": np.array(labels, dtype=np.int64),
    }


def prepare_dataset_salt(
    example, sentence_to_prompt, feature_extractor, processor, p_prompt=0.0
):
    audio = example["source"]
    input_features = feature_extractor(
        audio, sampling_rate=16_000, do_normalize=True
    ).input_features[0]
    labels = processor.tokenizer(str(example["target"])).input_ids
    labels.insert(
        1, salt.constants.SALT_LANGUAGE_TOKENS_WHISPER[example["target.language"]]
    )
    prompt = sentence_to_prompt.get(example["target"], None)
    # Use np.random.random() instead of np.random.rand()
    if prompt and np.random.random() < p_prompt:
        labels = list(processor.get_prompt_ids(prompt)) + labels
    return {
        "input_features": input_features,
        "labels": np.array(labels, dtype=np.int64),
        "source.language": example["source.language"],
        "target.language": example["target.language"],
    }

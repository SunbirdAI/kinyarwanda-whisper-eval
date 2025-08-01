#!/usr/bin/env python3
"""
Generate configuration files for all language/hour combinations.
Creates organized subdirectories for each language:

configs/
├── sna/                    # Shona configs
├── ful/                    # Fulani configs
└── lin/                    # Lingala configs

Each directory contains:
- baseline.yaml             # No training, just evaluation
- train_1h.yaml             # 1 hour training
- train_10h.yaml            # 10 hour training
- ... up to train_500h.yaml # 500 hour training
"""

import os
import yaml
from pathlib import Path

# Language configurations
LANGUAGES = {
    "shona": {"code": "sna", "subset": "sna"},
    "fulani": {"code": "ful", "subset": "ful"},
    "lingala": {"code": "lin", "subset": "lin"},
    # 'kinyarwanda': {'code': 'kin', 'subset': 'audio_50h'},  # Uncomment to regenerate kin configs
}

# Hour targets to generate configs for
HOUR_TARGETS = [1, 10, 50, 100, 150, 200, 500]

# Base configuration template
BASE_CONFIG = {
    "dataset_name": "evie-8/afrivoices",
    "run_training": True,
    "use_wandb": False,
    "use_mlflow": True,
    "seed": 42,
}


def generate_config(language_name: str, language_info: dict, hours: int) -> dict:
    """Generate a config for a specific language and hour target"""
    config = BASE_CONFIG.copy()
    config.update(
        {
            "experiment_name": f"whisper-large-v3-{language_name}-{hours}h",
            "dataset_subset": language_info["subset"],
            "language_code": language_info["code"],
            "target_hours": hours,
        }
    )
    return config


def main():
    # Create configs directory if it doesn't exist
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    # Generate configs for each language in its own subdirectory
    for language_name, language_info in LANGUAGES.items():
        # Create language-specific subdirectory
        lang_dir = configs_dir / language_info["code"]
        lang_dir.mkdir(exist_ok=True)

        # Generate baseline config (no training)
        baseline_config = {
            "experiment_name": f"baseline_whisper_{language_name}",
            "dataset_name": "evie-8/afrivoices",
            "dataset_subset": language_info["subset"],
            "language_code": language_info["code"],
            "run_training": False,
            "use_wandb": False,
            "use_mlflow": True,
            "seed": 42,
        }

        baseline_filename = lang_dir / "baseline.yaml"
        with open(baseline_filename, "w") as f:
            yaml.dump(baseline_config, f, default_flow_style=False)
        print(f"Generated: {baseline_filename}")

        # Generate training configs for each hour combination
        for hours in HOUR_TARGETS:
            config = generate_config(language_name, language_info, hours)

            train_filename = lang_dir / f"train_{hours}h.yaml"
            with open(train_filename, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Generated: {train_filename}")

    print(f"\n✅ Generated {len(LANGUAGES) * (len(HOUR_TARGETS) + 1)} config files")
    print(
        f"📁 Organized in language subdirectories: {[lang_info['code'] for lang_info in LANGUAGES.values()]}"
    )
    print(
        "\n💡 Note: Kinyarwanda (kin) configs already exist and use a different dataset structure."
    )
    print(
        "   Uncomment 'kinyarwanda' in LANGUAGES dict if you want to regenerate them."
    )
    print("\nExample usage:")
    print("# Baseline evaluation (no training)")
    print("uv run python train.py --config configs/sna/baseline.yaml")
    print("uv run python train.py --config configs/ful/baseline.yaml")
    print("\n# Training experiments")
    print("uv run python train.py --config configs/sna/train_1h.yaml")
    print("uv run python train.py --config configs/ful/train_50h.yaml")
    print("uv run python train.py --config configs/lin/train_100h.yaml")
    print("\n# Evaluation")
    print("uv run python eval.py --model_path <model_path> --language_subset sna")


if __name__ == "__main__":
    main()

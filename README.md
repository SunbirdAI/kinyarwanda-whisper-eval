# Kinyarwanda Whisper Evaluation

Evaluating Whisper model performance on Kinyarwanda across different amounts of labeled audio.

---

## 📦 Install

```bash
git clone https://github.com/SunbirdAI/kinyarwanda-whisper-eval.git
cd kinyarwanda-whisper-eval
uv sync
```

Install [SALT](https://github.com/SunbirdAI/salt):

```bash
git clone https://github.com/SunbirdAI/salt.git
uv pip install -r salt/requirements.txt
```

Set up environment:

```bash
cp env_example .env
```

Fill in your `.env` with MLflow and Hugging Face credentials.

---

## 🚀 Usage

### Baseline evaluation (no training)

```bash
uv run python train.py --config configs/baseline.yaml
```

### Fine-tuning experiments

```bash
uv run python train.py --config configs/train_1h.yaml
uv run python train.py --config configs/train_10h.yaml
uv run python train.py --config configs/train_50h.yaml
uv run python train.py --config configs/train_500h.yaml
```

### Evaluation

```bash
uv run python eval.py --model_path <model_path_or_hf_id> --batch_size=8
```

---

## 📁 Training Configs

| Config             | Hours  | Model ID on Hugging Face            |
| ------------------ | ------ | ----------------------------------- |
| `baseline.yaml`    | 0      | openai/whisper-large-v3             |
| `train_1h.yaml`    | 1      | akera/whisper-large-v3-kin-1h-v2    |
| `train_50h.yaml`   | 50     | akera/whisper-large-v3-kin-50h-v2   |
| `train_100h.yaml`  | 100    | akera/whisper-large-v3-kin-100h-v2  |
| `train_150h.yaml`  | 150    | akera/whisper-large-v3-kin-150h-v2  |
| `train_200h.yaml`  | 200    | akera/whisper-large-v3-kin-200h-v2  |
| `train_500h.yaml`  | 500    | akera/whisper-large-v3-kin-500h-v2  |
| `train_1000h.yaml` | 1000   | akera/whisper-large-v3-kin-1000h-v2 |
| `train_full.yaml`  | \~1400 | akera/whisper-large-v3-kin-full     |

Explore the collection:
👉 [https://huggingface.co/collections/Sunbird/kinyarwanda-hackathon-68872541c41c5d166d9bffad](https://huggingface.co/collections/Sunbird/kinyarwanda-hackathon-68872541c41c5d166d9bffad)

---

## 📊 Results

Evaluation on `dev_test[:300]` subset:

| Model                                 | Hours  | WER (%) | CER (%) | Score |
| ------------------------------------- | ------ | ------- | ------- | ----- |
| `openai/whisper-large-v3`             | 0      | 33.10   | 9.80    | 0.861 |
| `akera/whisper-large-v3-kin-1h-v2`    | 1      | 47.63   | 16.97   | 0.754 |
| `akera/whisper-large-v3-kin-50h-v2`   | 50     | 12.51   | 3.31    | 0.932 |
| `akera/whisper-large-v3-kin-100h-v2`  | 100    | 10.90   | 2.84    | 0.943 |
| `akera/whisper-large-v3-kin-150h-v2`  | 150    | 10.21   | 2.64    | 0.948 |
| `akera/whisper-large-v3-kin-200h-v2`  | 200    | 9.82    | 2.56    | 0.951 |
| `akera/whisper-large-v3-kin-500h-v2`  | 500    | 8.24    | 2.15    | 0.963 |
| `akera/whisper-large-v3-kin-1000h-v2` | 1000   | 7.65    | 1.98    | 0.967 |
| `akera/whisper-large-v3-kin-full`     | \~1400 | 7.14    | 1.88    | 0.970 |

> Score = 1 - (0.6 × CER + 0.4 × WER)

---

## 📒 Notes

* Metrics and training logs are tracked via MLflow.
* Evaluation uses fast `model.generate` with beam search over 300 samples.

---

## 📤 License

MIT

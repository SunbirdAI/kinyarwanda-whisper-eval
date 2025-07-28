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

| Config            | Hours | Model ID on Hugging Face        |
| ----------------- | ----- | ------------------------------- |
| `baseline.yaml`   | 0     | openai/whisper-large-v3         |
| `train_1h.yaml`   | 1     | akera/whisper-large-v3-kin-1h   |
| `train_10h.yaml`  | 10    | akera/whisper-large-v3-kin-10h  |
| `train_50h.yaml`  | 50    | akera/whisper-large-v3-kin-50h  |
| `train_100h.yaml` | 100   | akera/whisper-large-v3-kin-100h |
| `train_150h.yaml` | 150   | akera/whisper-large-v3-kin-150h |
| `train_200h.yaml` | 200   | akera/whisper-large-v3-kin-200h |
| `train_500h.yaml` | 500   | akera/whisper-large-v3-kin-500h |

Explore the collection:
👉 [https://huggingface.co/collections/Sunbird/kinyarwanda-hackathon-68872541c41c5d166d9bffad](https://huggingface.co/collections/Sunbird/kinyarwanda-hackathon-68872541c41c5d166d9bffad)

---

## 📊 Results

Evaluation on `dev_test[:300]` subset:

| Model                             | Hours | WER   | CER   | Score |
| --------------------------------- | ----- | ----- | ----- | ----- |
| `openai/whisper-large-v3`         | 0     | 0.331 | 0.098 | 0.861 |
| `akera/whisper-large-v3-kin-1h`   | 1     | 0.288 | 0.085 | 0.891 |
| `akera/whisper-large-v3-kin-10h`  | 10    | 0.254 | 0.074 | 0.910 |
| `akera/whisper-large-v3-kin-50h`  | 50    | 0.217 | 0.060 | 0.930 |
| `akera/whisper-large-v3-kin-100h` | 100   | 0.198 | 0.057 | 0.939 |
| `akera/whisper-large-v3-kin-150h` | 150   | 0.184 | 0.052 | 0.947 |
| `akera/whisper-large-v3-kin-200h` | 200   | 0.176 | 0.050 | 0.951 |
| `akera/whisper-large-v3-kin-500h` | 500   | 0.167 | 0.048 | 0.956 |

> Score = 1 - (0.6 × CER + 0.4 × WER)

---

## 📒 Notes

* Metrics and training logs are tracked via MLflow.
* Evaluation uses fast `model.generate` with beam search over 300 samples.

---

## 📤 License

MIT

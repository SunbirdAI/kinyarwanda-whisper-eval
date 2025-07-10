# kinyarwanda-whisper-eval
Evaluating Whisper performance wrt hours of labelled kinyarwanda audio


# Clone and install
```
git clone <your-repo-url>
cd kinyarwanda-whisper-eval
uv sync
```

# Install SALT framework  
```
git clone https://github.com/SunbirdAI/salt.git
uv pip install -r salt/requirements.txt
```

# Set up environment
```bash
cp env_example .env
```

# Edit .env with your MLflow and HF credentials


# Usage

### Baseline evaluation (no training)
```bash
uv run python train.py --config configs/baseline.yaml
```
#### Training experiments
```
uv run python train.py --config configs/train_1h.yaml
uv run python train.py --config configs/train_10h.yaml
uv run python train.py --config configs/train_20h.yaml
uv run python train.py --config configs/train_50h.yaml
```
# etc...


### Available Configs

```
baseline.yaml - Evaluate pre-trained Whisper (no training)
train_1h.yaml - Fine-tune on 1 hour of data
train_10h.yaml - Fine-tune on 10 hours of data
train_20h.yaml - Fine-tune on 20 hours of data
train_50h.yaml - Fine-tune on 50 hours of data
train_100h.yaml - Fine-tune on 100 hours of data
train_150h.yaml - Fine-tune on 150 hours of data
train_200h.yaml - Fine-tune on 200 hours of data
train_500h.yaml - Fine-tune on 500 hours of data
```

Metrics

Results include:

```
Final WER/CER: Word/Character Error Rate with normalization
Final Score: Evaluation score (higher is better)
Training metrics logged to MLflow
```
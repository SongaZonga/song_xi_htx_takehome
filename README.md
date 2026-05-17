# Fine-Tuning & Model Promotion Pipeline

Part B – Option 2 | LLM/LMM AI R&D Team Take-Home Assignment

---

## What It Does

This pipeline fine-tunes a DistilBERT model on the IMDb sentiment classification dataset and promotes the trained model only if it improves on both accuracy and AUC relative to the pre-training baseline. The pipeline is fully reproducible: running it twice with the same seed on the same hardware yields identical results.

**Pipeline stages:**

1. **Mount Google Drive** : All outputs (model weights, logs, metrics) are persisted to `/content/drive/MyDrive/final_model/` so they survive Colab session resets. The HuggingFace cache is also stored on Drive so the dataset and model weights are only downloaded once.
2. **Load & version dataset** : Downloads the IMDb dataset from the HuggingFace Hub with `revision="main"`, pinning the dataset to a specific upstream commit for reproducibility.
3. **Baseline evaluation** : Evaluates the pre-trained (unfine-tuned) model before any training begins, capturing baseline accuracy and AUC to use as the promotion comparison point.
4. **Fine-tune** : Fine-tunes `distilbert-base-uncased` for binary sentiment classification using the HuggingFace `Trainer` API across 3 epochs with a batch size of 32. Input sequences are truncated and padded to a max length of 256 tokens.
5. **Log metrics** : A custom `StepLoggingCallback` writes structured JSON logs (step, epoch, loss, learning rate, eval accuracy, eval AUC) to `training.log` every 50 steps. Final baseline and fine-tuned metrics are written together with improvement deltas to `metrics.json`.
6. **Promotion check** : `promotion.py` compares the fine-tuned model's accuracy and AUC against the pre-training baseline. The model is saved to Drive only if both metrics improve; otherwise it is discarded and the reason is recorded in `metrics.json`.

---

## How to Reproduce

> **Environment:** This notebook must be run in **Google Colab**. Do not run it in a local Jupyter environment — it depends on Colab's GPU runtime and Google Drive integration.

### 1. Open the notebook in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SongaZonga/song_xi_htx_takehome/blob/main/pipeline.ipynb)

Click the badge above, or paste this URL into your browser:

```
https://colab.research.google.com/github/SongaZonga/song_xi_htx_takehome/blob/main/pipeline.ipynb
```

This loads `pipeline.ipynb` directly from GitHub into Colab as an executable notebook. **Do not clone the repo first and try to open the notebook from the cloned files** — Colab cannot open notebooks from its local filesystem as new executable sessions.

Once the notebook is open, set the runtime to **GPU** (Runtime → Change runtime type → T4 GPU).

### 2. Clone the repository from inside the notebook

The first cell in the notebook clones the repo so that `promotion.py` is available for import:

```python
!git clone https://github.com/SongaZonga/song_xi_htx_takehome.git
%cd song_xi_htx_takehome
```

### 3. Mount Google Drive

The next cell mounts your Google Drive. Approve the prompt when asked. All outputs will be written under `/content/drive/MyDrive/final_model/`.

### 4. Run the notebook

Run all cells top to bottom. The seed is fixed at `42` throughout — no configuration needed.

### 5. Check outputs

All outputs are written to `/content/drive/MyDrive/final_model/`:

| File | Description |
|---|---|
| `metrics.json` | Full run log: baseline metrics, final metrics, improvement deltas, and promotion decision with reason |
| `training.log` | Structured JSON log written every 50 steps (loss, LR, eval accuracy, eval AUC per step/epoch) |
| `logs/` | HuggingFace Trainer logging directory |
| `model/` | Saved model weights and tokeniser — **only written if the model is promoted** |

### 6. Run the unit tests

```bash
uv run pytest tests/
```

---

## Design Decisions

**Training via Google Colab**
Google Colab was used due to initial hardware limitations where training on my local Nvdia rtx2060 GPU would take about 1 hour to complete. Therefore I opted to use Colab with an A100 GPU, which reduced the training time to 9 mins.

**Dataset versioning via `revision="main"`**
`datasets.load_dataset("imdb", revision="main")` pins the dataset to a specific upstream commit on the HuggingFace Hub. This is lightweight but sufficient — IMDb is a static benchmark that does not change. Storing the HuggingFace cache on Drive means the dataset is only fetched once and is available across Colab sessions without re-downloading.

**DistilBERT over BERT-base**
DistilBERT is ~40% smaller and ~60% faster than BERT-base with minimal accuracy loss on classification tasks. This keeps training feasible within a Colab session while still exercising a realistic fine-tuning workflow.

**Determinism via `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG`**
Beyond seeding `random`, `numpy`, and `torch`, the pipeline sets `torch.use_deterministic_algorithms(True)` and the `CUBLAS_WORKSPACE_CONFIG=:4096:8` environment variable. These force CUDA operations to use deterministic (non-approximate) kernels, which is required for full bit-level reproducibility on GPU. `cudnn.benchmark` is also disabled to prevent auto-tuning from selecting different kernels across runs.

**Promotion Metrics**
Accuracy is used as the primary promotion gate due to its interpretable nature and the balanced nature of the dataset. However, this alone may not indicate the confidence of prediction in the model. AUC however, complements this by checking whether the model has correctly learned to rank positive examples above negative ones across all thresholds. Next, requiring the fine-tuned model to beat the pre-training baseline on both Accuracy and AUC ensures the training actually imrpoved the model.

**Step-level logging via `StepLoggingCallback`**
Rather than relying solely on the Trainer's default logging, a custom callback writes structured JSON to `training.log` every 50 steps. This gives a granular audit trail of training dynamics (loss curves, LR schedule, per-epoch eval scores) that is queryable and easy to parse without a dedicated tracking server.

---

## One thing I would change with more time

**CI tests for the promotion check**
Currently the unit tests in `tests/` only run when someone manually invokes them. This is risky as `should_promote` decides whether a fine-tuned model gets saved. An improvement here would be to include a GitHub Actions workflow that runs the tests on every push and pull request.


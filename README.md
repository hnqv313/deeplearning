# Continual Image Classification without Catastrophic Forgetting

Reproducible class-incremental comparison of Naive Fine-tuning, EWC, LwF,
Replay, Frozen ViT-Tiny + NCM, and cumulative Joint fine-tuning as an
approximate offline upper bound.

## Protocol

| Stage | New classes | Evaluated classes |
|---|---|---|
| 0 | Dog, Cat | Dog, Cat |
| 1 | Car | Dog, Cat, Car |
| 2 | Person | Dog, Cat, Car, Person |
| 3 | Building | All five classes |

All methods use the same `vit_tiny_patch16_224` backbone, manifests,
augmentations, class order, and evaluation code. NCM freezes the backbone; all
gradient-based strategies fine-tune it. The model uses a fixed five-output head
and masks unseen logits, so no future images or labels participate in training.

## Environment

This project does not require an API key. Never place tokens, service-account
files, or credentials in source code, notebooks, YAML configs, manifests, or
experiment logs. Local `.env`, private-key, and common credential files are
ignored by Git; use environment variables if an optional external service is
added later, and share only a key-free `.env.example`.

New runs record the installed `timm` version and the backbone's pretrained
weight tag in `environment.json`. The completed Colab runs predate this field;
their exact `timm` version cannot be recovered from the saved artifacts.

For training:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

FiftyOne 1.22 supports Python 3.10-3.13. Install the dataset extras inside the
same environment:

```powershell
pip install -r requirements-data.txt
```

## 1. Download Open Images V7

Open Images is downloaded only through its official FiftyOne integration. The
raw download is intentionally larger than the final subset because crop filters
and class balancing happen afterward.

```powershell
python scripts/download_openimages.py --splits validation test --sampling-mode balanced --max-samples 1200
```

`balanced` requests a quota for each class separately, then merges duplicate
source images. This avoids the severe imbalance of a mixed query (in the pilot,
500 mixed images contained 333 Car annotations but only 10 Cat annotations).
With both source splits, the command downloads at most 12,000 source images.

## 2. Build balanced classification crops

```powershell
python scripts/build_crops.py --raw-manifest data/raw/manifests/validation.jsonl data/raw/manifests/test.jsonl --output-root data/processed_clean --output-manifest data/manifests/pool_clean.csv --target-per-class 500
python scripts/deduplicate.py data/manifests/pool_clean.csv
python scripts/partition_manifest.py --pool-manifest data/manifests/pool_clean.csv
```

The official validation and test sources are pooled and repartitioned by
`original_image_id` because FiftyOne's train split metadata contains roughly
nine million rows and can exceed the RAM available on student machines. This is
a custom course dataset, not an evaluation of the official Open Images test
split. Each source image contributes at most one crop, even if
it contains multiple target categories. Group boxes, depictions, inside views,
tiny boxes, and missing files are rejected.

For the visually difficult classes, the builder also uses available child
annotations. `Man`, `Woman`, `Boy`, and `Girl` map to Person; `House`,
`Office building`, `Skyscraper`, `Tower`, and `Castle` map to Building. Person
boxes marked occluded or truncated are rejected; Building boxes marked
occluded are rejected. The final balanced subset contains 500 crops per class.

## 3. Audit and verify

```powershell
python scripts/audit_dataset.py data/manifests/train.csv data/manifests/val.csv data/manifests/test.csv --output-dir outputs/dataset_audit_clean
python scripts/verify_dataset.py data/manifests/train.csv data/manifests/val.csv data/manifests/test.csv
```

Inspect all contact sheets under `outputs/dataset_audit_clean/` before training. Use
`deduplicate.py --apply` only after reviewing its duplicate report, then rebuild
or rebalance manifests if samples were removed.

## 4. Run experiments

One seed:

```powershell
python -m continual_dl.run --strategy naive --seed 42
python -m continual_dl.run --strategy joint --seed 42
python -m continual_dl.run --strategy ncm --seed 42
python -m continual_dl.run --strategy replay --seed 42
python -m continual_dl.run --strategy lwf --seed 42
python -m continual_dl.run --strategy ewc --seed 42
```

All configured seeds:

```powershell
python -m continual_dl.run --strategy replay --all-seeds
```

Every stage writes a JSON result, checkpoint, and confusion matrix under
`outputs/<strategy>/seed_<seed>/`.

The default uses `num_workers: 0` for reliable Windows execution. On Linux or
Colab, increase it to 2-4 after confirming the available CPU/RAM.

This machine currently has a CPU-only PyTorch build. The complete frozen
ViT-Tiny + NCM baseline is practical on CPU, but full fine-tuning for Naive,
Replay, LwF, EWC, and Joint should be run on a CUDA GPU. Copy the repository
and `data/processed_clean` to the GPU machine, install a CUDA-enabled PyTorch
build, verify `torch.cuda.is_available()` is `True`, then use the same commands
above. Pass `--device cuda` to fail early instead of silently falling back to
CPU.

On the GPU machine, run the complete method/seed matrix with:

```powershell
python scripts/run_benchmark.py --device cuda
```

The launcher resumes safely by skipping any run that already has a
`summary.json`. Preview its 15-run matrix without training using `--dry-run`;
use `--force` only when intentionally replacing completed results.

For Google Colab, use [`colab_train.ipynb`](colab_train.ipynb) and follow
[`COLAB.md`](COLAB.md). The notebook writes results directly to Google Drive so
an interrupted runtime can continue without repeating completed runs.

After all methods and seeds finish (or after copying the complete Colab output
tree to `outputs/colab_runs`):

```powershell
python scripts/summarize_results.py --outputs outputs/colab_runs
```

This produces `outputs/comparison.csv`, `outputs/per_seed_results.csv` and
`outputs/stage_accuracy.csv`, including peak allocated GPU memory.

Generate and validate the report figures with:

```powershell
.venv\Scripts\python.exe scripts\plot_final_results.py
```

This checks the full 6-method × 3-seed × 4-stage matrix and writes five PNG
figures under `outputs/final_figures/`.

## 5. NCM backbone initialization comparison

The published benchmark initializes every method from ImageNet-21k weights.
These three commands measure how much of the frozen-feature NCM result comes
from those weights, by re-running NCM on two other backbones. All other
settings — data, class order, augmentations, evaluation code and seeds — stay
identical, and results go to separate output trees so the published run stays
untouched.

First, pre-train a backbone from random initialization on the Stage 0 classes
only (`dog` and `cat`), using the same Naive strategy. NCM never trains the
backbone, so this stage is what gives it something to extract features with:

```powershell
python -m continual_dl.run --strategy naive --common-config configs/scratch_pretrain.yaml --all-seeds
```

Then run NCM on that backbone across all four stages, and on a never-trained
backbone as the floor:

```powershell
python -m continual_dl.run --strategy ncm --common-config configs/scratch.yaml --strategy-config configs/ncm_scratch_init.yaml --all-seeds
python -m continual_dl.run --strategy ncm --common-config configs/scratch_random.yaml --all-seeds
```

`configs/ncm_scratch_init.yaml` expands `{seed}` in `init_checkpoint`, so every
seed loads the checkpoint produced by the matching pre-training seed. Only the
`backbone.*` tensors are read; the 5-way head is discarded because NCM predicts
from class prototypes.

Summarize each arm separately, then build the comparison figures:

```powershell
python scripts/summarize_results.py --outputs outputs/scratch --output-csv outputs/scratch/comparison.csv
python scripts/summarize_results.py --outputs outputs/scratch_random --output-csv outputs/scratch_random/comparison.csv
python scripts/plot_pretrain_comparison.py
```

This writes four PNG figures under `outputs/pretrain_figures/`. The runtime
figure adds the Stage 0 pre-training cost to that arm, because the NCM stage
alone does not include it.

Pre-training is cheap in-domain supervision, not the absence of pre-training: it
uses 800 images of two classes for 500 optimizer steps, against 21k classes and
roughly 300M images for ImageNet-21k. That arm reaches 25.47% final accuracy
against 92.67% for the ImageNet-21k backbone, so the published NCM result does
not follow from the NCM rule alone. It does not show that a ViT cannot be trained
from scratch either, because the recipe used here is the fine-tuning one. See
`RESULTS.md` for the full breakdown and the limits of this comparison.

## Current verified status

- Final clean dataset: 2,500 crops, exactly 500 per class.
- Split: 2,000 train, 250 validation, 250 test; no missing files, duplicate
  hash leakage, or source-image leakage across splits.
- All six strategy paths complete the four-stage smoke test.
- The final Colab matrix is complete: six methods × three seeds = 18 T4 runs.
- All 18 summaries and 72 stage JSON files are mirrored in
  `outputs/colab_runs`; aggregate CSVs are generated directly from them.
- Frozen ViT-Tiny + NCM reaches 92.67% final accuracy, Replay reaches 91.20%,
  and the approximate Joint upper bound reaches 95.20%.
- The backbone initialization comparison is complete: three seeds per arm. NCM
  over a Stage 0 pre-trained backbone reaches 25.47% and over an untrained
  backbone 23.47%, against 92.67% for the ImageNet-21k backbone.
- See [`RESULTS.md`](RESULTS.md) for complete mean ± standard deviation tables,
  stage curves, timing and interpretation.

NCM is a strong low-compute baseline, not a universal fix. Freezing the
feature extractor removes parameter drift and prototype storage is tiny, but
feature extraction still costs time proportional to the number of new images,
and frozen ImageNet features may not adapt well to a different visual domain.

## Metric definitions

Metrics are computed from per-class accuracy histories. Final average accuracy
is the mean of the five final class accuracies. Average incremental accuracy is
the mean of the seen-class average accuracies after all four stages.

For each class evaluated at least twice, forgetting is its maximum observed
accuracy (including the final stage) minus final accuracy. Backward transfer is
final accuracy minus accuracy when that class first appeared. The final values
average over Dog, Cat, Car and Person. These are class-level variants, not the
task-level definitions used by many continual-learning papers; Dog and Cat
therefore contribute separately even though both arrive in Stage 0.

## 6. Tests

The tests use generated 32×32 images and the local `tiny_cnn` smoke-test
backbone; they do not download pretrained weights or datasets.

```powershell
pytest
```

## Fair-comparison rules

- Tune strategy hyperparameters on validation data only.
- Run seeds 42, 123, and 2026 and report mean ± standard deviation.
- Never add old images to EWC, LwF, Naive, or NCM.
- Replay has a fixed 200-image budget across all stages.
- Joint continues fine-tuning the same model on all data seen so far. It is an
  approximate offline upper bound, not a continual method or a retrain-from-
  scratch oracle.
- Report accuracy, forgetting, backward transfer, auxiliary memory, wall time,
  and peak GPU memory together.

# Final experiment results

## Protocol and environment

The final benchmark contains six methods, three random seeds (42, 123 and
2026), and four class-incremental stages. All 18 runs used the same Google
Colab Tesla T4 runtime class, dataset manifests, class order, ViT-Tiny model
family and evaluation code. The Joint baseline continues fine-tuning the same
model on all data seen so far; it is an approximate offline upper bound and is
not a valid continual-learning method.

The final dataset contains 2,500 Open Images crops: 500 each for Dog, Cat, Car,
Person and Building. The split contains 2,000 training, 250 validation and 250
test images, with no duplicate-hash or source-image leakage across splits.

## Aggregate comparison

Values are mean ± sample standard deviation across the three seeds. Accuracy,
forgetting and backward transfer are shown as percentages; time is seconds per
complete four-stage run. GPU memory is the maximum PyTorch-allocated memory
reported by `torch.cuda.max_memory_allocated` across the four stages. It does
not include all CUDA-reserved memory or other processes on the GPU.

| Method | Final accuracy | Avg. incremental accuracy | Forgetting | Backward transfer | Time | Peak GPU memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 20.00 ± 0.00 | 43.69 ± 1.17 | 96.67 ± 1.04 | -96.67 ± 1.04 | 501.36 ± 4.43 | 645.78 ± 0.00 |
| EWC | 20.00 ± 0.00 | 43.50 ± 0.90 | 96.83 ± 0.76 | -96.83 ± 0.76 | 551.75 ± 3.99 | 1039.38 ± 0.00 |
| LwF | 20.00 ± 0.00 | 48.93 ± 6.12 | 97.67 ± 0.76 | -97.67 ± 0.76 | 490.55 ± 3.54 | 682.81 ± 0.00 |
| Replay | 91.20 ± 5.01 | 94.11 ± 0.85 | 8.50 ± 5.57 | -8.17 ± 5.35 | 762.64 ± 5.90 | 645.78 ± 0.00 |
| Frozen ViT-Tiny + NCM | 92.67 ± 0.46 | 92.83 ± 0.19 | 2.67 ± 0.58 | -2.67 ± 0.58 | 42.05 ± 2.62 | 104.16 ± 0.00 |
| Joint (approx. upper bound) | 95.20 ± 1.39 | 95.84 ± 0.70 | 3.00 ± 1.80 | -1.83 ± 1.53 | 1247.07 ± 13.54 | 645.78 ± 0.00 |

Machine-readable values are stored in `outputs/comparison.csv`,
`outputs/per_seed_results.csv` and `outputs/stage_accuracy.csv`. All three are
generated directly from the 18 original `outputs/colab_runs/*/seed_*/summary.json`
files.

## Metric definitions used by this project

Let `a[k,j]` be the accuracy of class `j` after stage `k`. Final average
accuracy is the unweighted mean of the five class accuracies after Stage 3.
Average incremental accuracy is the mean of the seen-class average accuracies
after Stages 0, 1, 2 and 3.

Forgetting and backward transfer are implemented per class rather than per
task. For every class evaluated at least twice (Dog, Cat, Car and Person), the
code computes:

- forgetting: `max_k a[k,j] - a[3,j]`, where the maximum includes the final
  evaluation;
- backward transfer: `a[3,j] - a[first,j]`, where `first` is the stage in
  which class `j` first appears.

The reported values average these class-level quantities. Because Dog and Cat
both arrive in Stage 0, that stage contributes two class histories. These
definitions differ from the task-level definitions commonly used in the
continual-learning literature.

## Accuracy through the stream

| Method | Stage 0: Dog/Cat | Stage 1: +Car | Stage 2: +Person | Stage 3: +Building |
|---|---:|---:|---:|---:|
| Naive | 93.33% | 36.44% | 25.00% | 20.00% |
| EWC | 93.67% | 35.33% | 25.00% | 20.00% |
| LwF | 95.33% | 54.89% | 25.50% | 20.00% |
| Replay | 96.67% | 95.56% | 93.00% | 91.20% |
| Frozen ViT-Tiny + NCM | 91.00% | 94.00% | 93.67% | 92.67% |
| Joint (approx. upper bound) | 95.33% | 97.33% | 95.50% | 95.20% |

Report-ready figures are generated under `outputs/final_figures/`:

- `stage_accuracy.png`: mean accuracy after each incremental stage;
- `final_accuracy.png`: final five-class accuracy;
- `forgetting_and_bwt.png`: forgetting and backward transfer;
- `runtime.png`: wall-clock cost on the common Colab GPU;
- `gpu_memory.png`: peak PyTorch-allocated GPU memory.

Regenerate and validate them with:

```powershell
.venv\Scripts\python.exe scripts\plot_final_results.py
```

## Interpretation

Naive, EWC and LwF all finish at chance-level accuracy (20%) after the fifth
class arrives. Under the tested hyperparameters, neither parameter importance
regularization nor output distillation prevents the single-head classifier
from predicting only the newest class after Stage 3. Representation drift was
not measured directly. At the Car stage, LwF ranges from 36.67% to 80.00%; for
seed 2026, 29 of the 30 Dog/Cat errors are predictions of Car.

Replay is the strongest gradient-based continual method. It reaches 91.20%
final accuracy, but its 5.01-point seed standard deviation is substantially
larger than NCM's 0.46 points. The fixed 200-image memory therefore works well
but its seed-to-seed variance is large. The current runs cannot separate the
effect of buffer composition from GPU-training nondeterminism.

Frozen ViT-Tiny + NCM reaches 92.67% final accuracy, 2.53 percentage points
below the approximate Joint upper bound and 1.47 points above Replay. The
1.47-point difference is smaller than Replay's 5.01-point seed standard
deviation, and the ranking reverses across seeds, so NCM and Replay should be
treated as comparable in accuracy. Under the configured epoch and optimizer-
step budgets, NCM is about 18 times faster than Replay and 30 times faster than
Joint. Its 104.16 MiB peak allocated GPU memory is about 84% lower than
Replay's 645.78 MiB and about 90% lower than EWC's 1039.38 MiB. Its low
forgetting is consistent with freezing the representation and retaining class
prototypes rather than repeatedly updating weights.

Peak GPU allocation is not the complete auxiliary-memory budget. Replay also
stores a fixed 200-image buffer, NCM stores one feature prototype per learned
class, EWC retains Fisher/parameter statistics and LwF retains a teacher model.
Those method-specific states may live partly outside the measured CUDA
allocation and must be described separately in the report.

The EWC peak of 1039.38 MiB already occurs at Stage 0, before an EWC penalty can
affect training. The implementation estimates the Fisher matrix after each
stage without AMP, so this peak is an implementation-specific cost rather than
an unavoidable memory requirement of EWC.

The five gradient-based methods produce different Stage-0 accuracies even for
the same seed, although their Stage-0 optimization is equivalent. For seed 42,
the range is 91% to 99%. The code enables deterministic cuDNN behavior but does
not call `torch.use_deterministic_algorithms(True)`; AMP and two data-loader
workers may also contribute. Differences near the observed run-to-run noise
should therefore not be interpreted as algorithmic effects.

NCM prototypes are computed once from the training transform, which includes
random crop, horizontal flip and color jitter. This is a source of variation
between NCM seeds even though its backbone is frozen.

All reported runs use one fixed configuration per strategy. The repository
contains no hyperparameter sweep or tuning log; only the group can confirm
whether any informal tuning occurred outside the recorded workflow. The exact
Colab `timm` version and pretrained-weight tag were not captured, so they
cannot be recovered reliably from the completed artifacts. Future runs now
record both fields in `environment.json`.

These results support NCM as a strong low-compute baseline for this dataset,
not as a universal solution. Feature extraction still scales with the number
of incoming images; performance depends on ImageNet-pretrained features; and
the method may fail under large domain shift, fine-grained classes or changing
class distributions.

## Artifact completeness

The complete 18-run directory has been mirrored locally under
`outputs/colab_runs`. It contains 18 summaries and 72 stage JSON files. The
aggregate tables, stage trajectories and GPU-memory values are regenerated
from those JSON files rather than transcribed from the Colab display.

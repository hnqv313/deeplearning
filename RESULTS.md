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

## Backbone initialization comparison

NCM freezes the backbone and computes prototypes from its features, so its result
depends entirely on where those features came from. Three arms share the same
data, class order, augmentations, evaluation code and seeds, and differ only in
the backbone:

| Arm | Backbone | Final accuracy (%) | Avg. incremental (%) | Forgetting (%) | BWT (%) | Time (s) | Peak GPU (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| ImageNet-21k pretrained | timm default weights | 92.67 ± 0.46 | 92.83 ± 0.19 | 2.67 ± 0.58 | −2.67 ± 0.58 | 42.05 ± 2.62 | 104.16 |
| Pre-trained on Stage 0 only | random init, then Naive on Dog and Cat | 25.47 ± 1.40 | 39.06 ± 2.35 | 21.83 ± 4.80 | −21.83 ± 4.80 | 236.6 | 644.67 |
| Untrained (random init) | random init, no training | 23.47 ± 1.01 | 35.52 ± 2.09 | 22.67 ± 2.02 | −22.67 ± 2.02 | 50.83 ± 0.49 | 103.16 |

Mean ± sample standard deviation over seeds 42, 123 and 2026. The Stage 0 time
includes its Naive pre-training run (185.4 s mean) plus its NCM stage (51.2 s); the other two figures cover
the NCM stage only. Peak GPU memory for the Stage 0 arm is likewise the maximum across both steps: the
Naive pre-training run needs 644.67 MiB while the NCM stage alone needs 103.16 MiB, so the arm cannot be run
in less memory than the pre-training step requires. The Stage 0 checkpoint is produced once and shared with
the Replay arm below, so the two pre-training costs are not additive.

Average accuracy on seen classes through the stream (%):

| Stage | Seen classes | Pretrained | Stage 0 pre-trained | Random init |
|---:|---|---:|---:|---:|
| 0 | 2 | 91.00 | 51.67 | 46.67 |
| 1 | 3 | 94.00 | 45.11 | 41.11 |
| 2 | 4 | 93.67 | 34.00 | 30.83 |
| 3 | 5 | 92.67 | 25.47 | 23.47 |

Per-seed final accuracy (%):

| Arm | 42 | 123 | 2026 |
|---|---:|---:|---:|
| Pretrained | 92.40 | 92.40 | 93.20 |
| Stage 0 pre-trained | 24.00 | 25.60 | 26.80 |
| Random init | 23.60 | 22.40 | 24.40 |

Machine-readable values are in `outputs/scratch/*.csv` and
`outputs/scratch_random/*.csv`; figures are in `outputs/pretrain_figures/`.

### What the comparison establishes

With the NCM rule, the data and the evaluation held fixed, only the ImageNet-21k
backbone reaches competitive accuracy. The two backbones without those weights
land near the 20% chance level of a five-class problem, so the published 92.67%
is not produced by the NCM rule itself. This is direct evidence for the
ImageNet-21k domain-advantage threat recorded in
`docs/report/LIMITATIONS_AND_THREATS.md` section 4.1.

### What the comparison does not establish

- **It is not a measurement of pre-training versus no pre-training.** The Stage 0
  arm is cheap in-domain supervision: 800 images, two classes, 500 optimizer
  steps. ImageNet-21k is 21k classes and roughly 300M images. The comparison
  measures the effect of that scale and domain match, nothing more.
- **The Stage 0 arm underfits, so it does not show that ViT from scratch fails.**
  Its own Naive run reaches only 57%, 58% and 59% Stage 0 test accuracy, against
  91% for pretrained NCM. With 800 images, 500 steps, AdamW at 1e-4, no warmup
  and no learning-rate schedule, the configuration is a fine-tuning recipe being
  applied to random initialization. A dedicated from-scratch recipe was not
  tuned, because tuning on validation data is reserved for the compared methods.
- **The 2.00-point gap over random init is inside the noise floor.** Per-seed
  Stage 0 values span 24.00 to 26.80 and random-init values span 22.40 to 24.40,
  so the two ranges overlap. With three seeds and a 250-image test set there is
  no statistical basis for calling the arms different.
- **The pretrained arm cannot be broken down per class.** Only the aggregate
  CSVs are retained for the published 18 runs; the raw `summary.json` files are
  not in the repository, so per-class accuracies are available for the two new
  arms only.
- **Class-level behaviour in the Stage 0 arm is not ordered as expected.** Its
  final per-class accuracies are Dog 11.33%, Cat 22.00%, Car 39.33%, Person
  18.67% and Building 36.00%. Dog is the worst class even though Dog was trained
  on, which is consistent with an underfitted feature extractor rather than with
  a useful representation of the pre-training classes.

## Replay on a non-pretrained backbone

The three arms above isolate the effect of the backbone source while holding the
method fixed at NCM, which never trains its backbone. Replay does train it, so it
answers a different question: whether the only gradient-based method that works
survives without ImageNet weights. It starts from the same Stage 0 checkpoint as
the NCM arm above, with `configs/replay.yaml` hyperparameters unchanged.

| Arm | Final accuracy (%) | Avg. incremental (%) | Forgetting (%) | BWT (%) | Time (s) | Peak GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Replay, ImageNet-21k backbone | 91.20 ± 5.01 | 94.11 ± 0.85 | 8.50 ± 5.57 | −8.17 ± 5.35 | 762.64 ± 5.90 | 645.78 |
| Replay, Stage 0 pre-trained backbone | 45.60 ± 4.33 | 56.33 ± 2.05 | 39.00 ± 5.89 | −36.33 ± 6.79 | 765.6 | 644.67 |

Average accuracy on seen classes through the stream (%):

| Stage | Seen classes | Replay, pretrained | Replay, Stage 0 pre-trained |
|---:|---|---:|---:|
| 0 | 2 | 96.67 | 56.67 |
| 1 | 3 | 95.56 | 64.22 |
| 2 | 4 | 93.00 | 58.83 |
| 3 | 5 | 91.20 | 45.60 |

Per-seed final accuracy (%): 50.40 (seed 42), 44.40 (seed 123), 42.00 (seed 2026),
against 96.00, 86.00 and 91.60 for the pretrained backbone. Per-seed forgetting (%):
32.5, 44.0 and 40.5, against 3.5, 14.5 and 7.5.

Final per-class accuracies of the Stage 0 arm (%): Building 82.00, Cat 41.33,
Dog 38.67, Person 36.67, Car 29.33.

### What this establishes

Rehearsal is markedly more robust than frozen prototypes when the backbone is
weak. Both arms start from the same checkpoint, so the 20.13-point difference
between Replay at 45.60% and NCM at 25.47% is attributable to fine-tuning with a
200-image memory rather than freezing the backbone. Both non-pretrained arms also
remain above EWC and LwF, which reach 20.00% while using ImageNet weights.

### What this does not establish

- **Replay does depend heavily on pretrained features.** It loses 45.60 points,
  more than half its accuracy, and its forgetting grows from 8.50% to 39.00%. So
  the benchmark's success is not solely an artefact of pre-training, but
  pre-training roughly doubles what the method achieves.
- **The 45.60% figure is a floor imposed by the recipe, not by Replay.** The
  configuration is the fine-tuning one: AdamW at 1e-4, 20 epochs on Stage 0 and
  15 afterwards, no warmup, no learning-rate schedule, 500 steps on Stage 0.
  Stage 0 accuracy of this arm is 56.67% against 96.67% for the pretrained
  backbone, so it starts from a much weaker model and never catches up. A
  from-scratch recipe was not tuned, because tuning is reserved on validation
  data for the compared methods.
- **The Stage 0 arms do not all trace back to one identical Stage 0 run.** The
  pre-training runs scored 57.00%, 58.00% and 59.00% on Stage 0, while this arm
  scored 55.00%, 58.00% and 57.00% from the same checkpoints and seed. At Stage 0
  Replay should be equivalent to Naive, so the 0 to 2 point gap is the GPU
  non-determinism already recorded in `LIMITATIONS` section 1.1.
- **Three seeds, no statistical test.** The 45.60% mean carries a 4.33 point
  standard deviation.

Reproduce with the commands in `README.md` section 5, then summarize each arm
into `outputs/scratch` and `outputs/scratch_random` and run
`scripts/plot_pretrain_comparison.py`.

## Replay + NCM hybrid

The hybrid trains the backbone with Replay, then rebuilds every prototype from
the memory buffer using that stage's backbone and predicts by cosine similarity.
It keeps Replay's representation learning and replaces the softmax head with
NCM's prototype head. Memory budget and ratio match Replay, so the two differ
only in the head.

| Backbone | Method | Final accuracy (%) | Avg. incremental (%) | Forgetting (%) | BWT (%) | Time (s) | Peak GPU (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| ImageNet-21k | Replay+NCM hybrid | 95.60 ± 1.06 | 95.50 ± 0.40 | 2.83 ± 1.89 | −1.50 ± 1.32 | 637.56 ± 23.82 | 644.67 |
| ImageNet-21k | Joint (upper bound) | 95.20 ± 1.39 | 95.84 ± 0.70 | 3.00 ± 1.80 | −1.83 ± 1.53 | 1247.07 ± 13.54 | 645.78 |
| ImageNet-21k | NCM | 92.67 ± 0.46 | 92.83 ± 0.19 | 2.67 ± 0.58 | −2.67 ± 0.58 | 42.05 ± 2.62 | 104.16 |
| ImageNet-21k | Replay | 91.20 ± 5.01 | 94.11 ± 0.85 | 8.50 ± 5.57 | −8.17 ± 5.35 | 762.64 ± 5.90 | 645.78 |
| Stage 0 | Replay+NCM hybrid | 53.47 ± 4.28 | 57.16 ± 1.70 | 14.00 ± 2.65 | −9.33 ± 3.69 | 724.96 ± 2.95 | 644.67 |
| Stage 0 | Replay | 45.60 ± 4.33 | 56.33 ± 2.05 | 39.00 ± 5.89 | −36.33 ± 6.79 | 580.20 ± 3.16 | 644.67 |
| Stage 0 | NCM | 25.47 ± 1.40 | 39.06 ± 2.35 | 21.83 ± 4.80 | −21.83 ± 4.80 | 51.17 ± 0.66 | 103.16 |

Average accuracy on seen classes through the stream (%):

| Stage | Seen classes | Hybrid, ImageNet | Hybrid, stage 0 | Replay, ImageNet | NCM, ImageNet |
|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 94.33 | 48.00 | 96.67 | 91.00 |
| 1 | 3 | 96.22 | 68.00 | 95.56 | 94.00 |
| 2 | 4 | 95.83 | 59.17 | 93.00 | 93.67 |
| 3 | 5 | 95.60 | 53.47 | 91.20 | 92.67 |

Per-seed final accuracy (%): hybrid on ImageNet 94.40, 96.40, 96.00; hybrid on the
Stage 0 backbone 54.40, 57.20, 48.80.

Final per-class accuracy (%):

| Backbone | Dog | Cat | Car | Person | Building |
|---|---:|---:|---:|---:|---:|
| ImageNet-21k | 93.33 | 96.00 | 95.33 | 96.67 | 96.67 |
| Stage 0 | 26.00 | 58.00 | 80.67 | 45.33 | 57.33 |

![Comparison](../../outputs/hybrid_figures/hybrid_backbone_comparison.png)
*Figure. Final accuracy and forgetting for NCM, Replay and the hybrid on both backbones.*

### What the hybrid establishes

**On the ImageNet-21k backbone it is the strongest method measured here, above
the Joint reference.** 95.60% against Joint's 95.20%, a 0.40-point difference
that is well inside both standard deviations, so the two are not separable. It
is separable from its parents: +2.93 points over NCM and +4.40 over Replay, and
the per-seed ranges do not overlap. It reaches that in 638 s, about half of
Joint's 1247 s, and its accuracy spread across seeds is 1.06 points against
Replay's 5.01.

**On the Stage 0 backbone it gains 7.87 points over Replay and cuts forgetting
by 25 points.** 53.47% against 45.60%, with forgetting falling from 39.00% to
14.00%. Both parents are clearly separated: NCM reaches 25.47% and Replay 45.60%.
This is the clearest single result of the comparison, and it confirms the
hypothesis the arm was built to test: on a weak backbone most of Replay's loss
came from the softmax head shifting towards newly added classes, not from the
representation itself.

**Joint's claim to be an upper bound does not hold for final accuracy.** The
hybrid exceeds it while being a valid continual method, because a cumulative
softmax head remains biased towards recently added classes even when all data is
available. The prototype head removes that bias.

### What it does not establish

- **The prototypes are noisier than NCM's.** The hybrid estimates each prototype
  from the 40 to 100 buffer images available at that stage, while the NCM arm uses
  all 400 training crops per class. The hybrid's advantage is the combination, not
  a better prototype estimate.
- **The 0.40-point lead over Joint is not significant.** Three seeds, no
  statistical test. The defensible claim is that the hybrid matches the Joint
  reference while using half the wall time and remaining a valid continual method.
- **Prototype refresh cost is not isolated.** Hybrid per-stage `training_seconds`
  runs about 25% above Replay's, but the two arms were run in different GPU
  sessions, and the refresh itself is 200 forward-only images per stage. The
  difference cannot be attributed to the refresh from these runs.
- **The hybrid inherits Replay's storage.** It keeps the same 200-image buffer;
  no new auxiliary memory is introduced.

## Artifact completeness

The complete 18-run directory has been mirrored locally under
`outputs/colab_runs`. It contains 18 summaries and 72 stage JSON files. The
aggregate tables, stage trajectories and GPU-memory values are regenerated
from those JSON files rather than transcribed from the Colab display.

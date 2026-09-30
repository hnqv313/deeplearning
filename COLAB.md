# Colab GPU runbook

Use [`colab_train.ipynb`](colab_train.ipynb) for the final experiment matrix.
It mounts Google Drive, extracts the key-free project bundle, verifies CUDA and
the dataset, stores results persistently in Drive, and launches all six methods
for seeds 42, 123, and 2026.

## What you do

1. Upload `continual_dl_colab_bundle.zip` to the root of `MyDrive`.
2. Upload or open `colab_train.ipynb` in Google Colab.
3. Select **Runtime > Change runtime type > T4 GPU** (or a faster GPU).
4. Run cells from top to bottom.
5. If Colab disconnects, reconnect and rerun the cells. Completed runs are
   skipped because each finished run has a `summary.json` in Drive.

Results are written to `MyDrive/continual_dl_outputs`. The Colab configuration
does not save model checkpoints, which avoids several gigabytes of unnecessary
Drive writes. It does retain per-stage metrics and confusion matrices needed
for the final report.

No API key is required. Google Drive authorization happens through Colab's
standard mount dialog; never paste tokens into notebook cells.

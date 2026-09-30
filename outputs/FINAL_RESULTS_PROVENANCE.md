# Final result provenance

- Source runtime: Google Colab, Tesla T4.
- Notebook: `colab_train.ipynb`.
- Completed matrix: six strategies × seeds 42, 123 and 2026 = 18 runs.
- Persistent source directory: `MyDrive/continual_dl_outputs`.
- Local raw mirror: `outputs/colab_runs`, containing 18 `summary.json`, 72
  `stage_*.json`, 18 `config.json` and 18 `environment.json` files.
- `comparison.csv`, `per_seed_results.csv` and `stage_accuracy.csv` are
  generated directly from the 18 original summary JSON files by
  `scripts/summarize_results.py`.
- Peak GPU memory is the maximum `peak_gpu_memory_bytes` across the four stages
  of each run, converted to MiB using 1 MiB = 1,048,576 bytes.
- `scripts/plot_final_results.py` validates strategy, seed and stage coverage
  before generating figures.

The original Drive directory was downloaded as
`continual_dl_outputs-20260929T182003Z-1-001.zip` and preserved at the project
root. Its SHA-256 is
`F192E0D17907912BDF18644D36EE5B4F2C7F04144C694C05A2ED9D38F3CFA09B`.
The imported raw files remain unchanged under `outputs/colab_runs`.

No API key, credential or secret was entered or created during synchronization.

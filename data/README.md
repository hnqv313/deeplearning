# Dataset directory

Generated files are intentionally not committed.

Expected manifests:

- `data/manifests/train.csv`
- `data/manifests/val.csv`
- `data/manifests/test.csv`

Each CSV contains at least:

```text
sample_id,original_image_id,filepath,label,label_id,source_split,bbox,sha256
```

Use the scripts in `scripts/` to download and build the Open Images V7 subset.

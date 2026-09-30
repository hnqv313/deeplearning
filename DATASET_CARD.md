# Dataset card: Open Images V7 continual subset

## Source

- Dataset: Open Images V7
- Official page: <https://storage.googleapis.com/openimages/web/download_v7.html>
- Target box classes: Dog, Cat, Car, Person, Building
- Intended task: single-label class-incremental image classification

Open Images images are distributed under their source licenses, commonly
Creative Commons Attribution. Preserve the Open Images image ID and follow the
official attribution requirements when publishing example images.

## Construction

The project downloads only images with target-class detection annotations. It
keeps at most one target crop per original source image, rejects group boxes,
depictions, inside views, very small boxes, and corrupt files, and applies 8%
context padding. Person additionally uses Man/Woman/Boy/Girl annotations and
rejects occluded or truncated boxes. Building additionally uses House/Office
building/Skyscraper/Tower/Castle annotations and rejects occluded boxes. The
target subset is balanced per class:

| Split | Per class | Total |
|---|---:|---:|
| Train | 400 | 2,000 |
| Validation | 50 | 250 |
| Test | 50 | 250 |

The official validation and test images are pooled, then repartitioned into the
course train/validation/test sets by original image ID. This avoids the
high-memory metadata pass required by the nine-million-image Open Images train
split. Exact and perceptual duplicate checks are run before partitioning. No
original image ID may occur in multiple course splits.

## Known limitations

- Object crops are easier than full-scene image classification.
- Bounding-box context and crop scale differ between object categories.
- Some valid Person crops contain only a clearly recognizable upper/lower body,
  and some Building crops depict entrances or interiors rather than facades.
- `Person` raises privacy and representation concerns; only public research
  images and the released Open Images annotations are used.
- ImageNet-pretrained backbones may already encode the five target concepts.
- A single `Building` prototype may be weak because the visual class is highly
  multimodal.

## Required manual audit

Before training, inspect every generated contact sheet under
`outputs/dataset_audit_clean/`. Remove mislabeled, unsafe, ambiguous, or low-quality
crops and restore class balance before producing final manifests.

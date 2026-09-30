"""Build balanced single-object classification crops from raw Open Images JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


CLASS_MAP = {
    "Dog": ("dog", 0),
    "Cat": ("cat", 1),
    "Car": ("car", 2),
    "Person": ("person", 3),
    "Man": ("person", 3),
    "Woman": ("person", 3),
    "Boy": ("person", 3),
    "Girl": ("person", 3),
    "Building": ("building", 4),
    "House": ("building", 4),
    "Office building": ("building", 4),
    "Skyscraper": ("building", 4),
    "Tower": ("building", 4),
    "Castle": ("building", 4),
}


@dataclass(frozen=True)
class Candidate:
    image_id: str
    filepath: Path
    source_split: str
    detection_index: int
    source_label: str
    bbox: tuple[float, float, float, float]


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _valid_detection(detection: dict[str, Any], class_name: str) -> bool:
    if any(
        _truthy(detection.get(key))
        for key in ("is_group_of", "is_depiction", "is_inside")
    ):
        return False
    if class_name == "person" and (
        _truthy(detection.get("is_occluded"))
        or _truthy(detection.get("is_truncated"))
    ):
        return False
    if class_name == "building" and _truthy(detection.get("is_occluded")):
        return False
    return True


def _quality_score(detection: dict[str, Any]) -> tuple[int, int, float]:
    bbox = detection["bounding_box"]
    return (
        int(not _truthy(detection.get("is_occluded"))),
        int(not _truthy(detection.get("is_truncated"))),
        float(bbox[2]) * float(bbox[3]),
    )


def load_candidates(paths: list[Path]) -> dict[str, list[Candidate]]:
    candidates: dict[str, list[Candidate]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                best_by_class: dict[str, tuple[int, str, dict[str, Any]]] = {}
                for index, detection in enumerate(record.get("detections", [])):
                    source_label = detection.get("label")
                    if source_label not in CLASS_MAP:
                        continue
                    class_name = CLASS_MAP[source_label][0]
                    if not _valid_detection(detection, class_name):
                        continue
                    bbox = detection.get("bounding_box")
                    if not bbox or len(bbox) != 4:
                        continue
                    previous = best_by_class.get(class_name)
                    if previous is None or _quality_score(detection) > _quality_score(previous[2]):
                        best_by_class[class_name] = (index, source_label, detection)

                for class_name, (index, source_label, detection) in best_by_class.items():
                    candidates[class_name].append(
                        Candidate(
                            image_id=str(record["image_id"]),
                            filepath=Path(record["filepath"]),
                            source_split=str(record["source_split"]),
                            detection_index=index,
                            source_label=source_label,
                            bbox=tuple(float(value) for value in detection["bounding_box"]),
                        )
                    )
    return candidates


def crop_candidate(
    candidate: Candidate,
    output_path: Path,
    padding: float,
    minimum_side: int,
    minimum_area_ratio: float,
) -> tuple[int, int] | None:
    if not candidate.filepath.exists():
        return None
    with Image.open(candidate.filepath) as source_handle:
        image = ImageOps.exif_transpose(source_handle).convert("RGB")
        width, height = image.size
        x, y, box_width, box_height = candidate.bbox
        if box_width * box_height < minimum_area_ratio:
            return None
        x0 = max(0.0, x - box_width * padding)
        y0 = max(0.0, y - box_height * padding)
        x1 = min(1.0, x + box_width * (1.0 + padding))
        y1 = min(1.0, y + box_height * (1.0 + padding))
        pixel_box = (
            int(round(x0 * width)),
            int(round(y0 * height)),
            int(round(x1 * width)),
            int(round(y1 * height)),
        )
        crop_width = pixel_box[2] - pixel_box[0]
        crop_height = pixel_box[3] - pixel_box[1]
        if min(crop_width, crop_height) < minimum_side:
            return None
        crop = image.crop(pixel_box)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path, format="JPEG", quality=95, optimize=True)
        return crop.size


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_split(
    raw_manifests: list[Path],
    output_root: Path,
    output_manifest: Path,
    target_per_class: int,
    seed: int,
    padding: float,
    minimum_side: int,
    minimum_area_ratio: float,
) -> None:
    candidates = load_candidates(raw_manifests)
    rng = random.Random(seed)
    for class_candidates in candidates.values():
        rng.shuffle(class_candidates)

    used_image_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    # Select constrained classes first so a multi-label source image cannot be
    # consumed by an abundant class before a rare class (Building/Cat) sees it.
    selection_order = sorted(
        set(CLASS_MAP.values()),
        key=lambda item: (len(candidates.get(item[0], [])), item[1]),
    )
    for class_name, label_id in selection_order:
        selected = 0
        for candidate in candidates.get(class_name, []):
            if candidate.image_id in used_image_ids:
                continue
            filename = (
                f"{candidate.source_split}_{class_name}_{candidate.image_id}_"
                f"{candidate.detection_index}.jpg"
            )
            output_path = output_root / candidate.source_split / class_name / filename
            size = crop_candidate(
                candidate,
                output_path,
                padding=padding,
                minimum_side=minimum_side,
                minimum_area_ratio=minimum_area_ratio,
            )
            if size is None:
                continue
            used_image_ids.add(candidate.image_id)
            sample_id = output_path.stem
            rows.append(
                {
                    "sample_id": sample_id,
                    "original_image_id": candidate.image_id,
                    "filepath": output_path.as_posix(),
                    "label": class_name,
                    "label_id": label_id,
                    "source_split": candidate.source_split,
                    "bbox": json.dumps(candidate.bbox),
                    "sha256": sha256(output_path),
                    "crop_width": size[0],
                    "crop_height": size[1],
                }
            )
            selected += 1
            if selected >= target_per_class:
                break
        if selected < target_per_class:
            raise RuntimeError(
                f"Only built {selected}/{target_per_class} crops for {class_name} from "
                f"{raw_manifests}. Increase --max-samples during download or relax crop filters."
            )

    rows.sort(key=lambda row: (int(row["label_id"]), str(row["sample_id"])))
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} crops and {output_manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--target-per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--minimum-side", type=int, default=80)
    parser.add_argument("--minimum-area-ratio", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_split(
        raw_manifests=args.raw_manifest,
        output_root=args.output_root,
        output_manifest=args.output_manifest,
        target_per_class=args.target_per_class,
        seed=args.seed,
        padding=args.padding,
        minimum_side=args.minimum_side,
        minimum_area_ratio=args.minimum_area_ratio,
    )


if __name__ == "__main__":
    main()

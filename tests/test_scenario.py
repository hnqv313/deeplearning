from torchvision.transforms import Compose, Resize, ToTensor

from continual_dl.constants import CLASS_TO_ID
from continual_dl.data.scenario import build_experiences


def test_build_experiences_filters_new_and_seen_classes(tiny_manifests):
    transform = Compose([Resize((32, 32)), ToTensor()])
    experiences = build_experiences(
        train_manifest=tiny_manifests["train"],
        val_manifest=tiny_manifests["val"],
        test_manifest=tiny_manifests["test"],
        stages=(("dog", "cat"), ("car",), ("person",), ("building",)),
        class_to_id=CLASS_TO_ID,
        train_transform=transform,
        eval_transform=transform,
    )
    assert len(experiences) == 4
    assert len(experiences[0].new_train_dataset) == 8
    assert len(experiences[1].new_train_dataset) == 4
    assert len(experiences[1].seen_train_dataset) == 12
    assert experiences[-1].seen_class_names == ("dog", "cat", "car", "person", "building")

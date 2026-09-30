"""Project-wide constants shared by data, training, and evaluation code."""

CLASS_ORDER = ("dog", "cat", "car", "person", "building")
OPEN_IMAGES_CLASS_NAMES = {
    "dog": "Dog",
    "cat": "Cat",
    "car": "Car",
    "person": "Person",
    "building": "Building",
}
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_ORDER)}
ID_TO_CLASS = {index: name for name, index in CLASS_TO_ID.items()}
DEFAULT_STAGES = (("dog", "cat"), ("car",), ("person",), ("building",))

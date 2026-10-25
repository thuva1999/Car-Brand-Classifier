
from datasets import load_dataset

print("Loading dataset information...")

dataset = load_dataset(
    "tanganke/stanford_cars",
    split="train",
    streaming=True
)

print("\nDataset features:")
print(dataset.features)

label_feature = dataset.features["label"]

print("\nLabel information:")
print(label_feature)

if hasattr(label_feature, "names") and label_feature.names:
    print("\nAvailable car classes:")
    for i, name in enumerate(label_feature.names):
        print(i, ":", name)

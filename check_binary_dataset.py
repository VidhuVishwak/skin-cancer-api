from pathlib import Path

root = Path("dataset_binary")

for split in ["train", "valid"]:
    print(f"\n--- {split.upper()} ---")
    for label in ["benign", "malignant"]:
        folder = root / split / label
        count = len(list(folder.glob("*.jpg")))
        print(f"{label}: {count}")
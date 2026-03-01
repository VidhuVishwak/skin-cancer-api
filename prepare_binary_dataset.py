import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Paths
metadata_path = "dataset/HAM10000_metadata.csv"
images_part1 = Path("dataset/HAM10000_images_part_1")
images_part2 = Path("dataset/HAM10000_images_part_2")
output_root = Path("dataset_binary")

# Create output folders
for split in ["train", "valid"]:
    for label in ["benign", "malignant"]:
        (output_root / split / label).mkdir(parents=True, exist_ok=True)

# Load metadata
df = pd.read_csv(metadata_path)

# Define malignant and benign classes
malignant_classes = ["mel", "bcc", "akiec"]

def label_binary(dx):
    return "malignant" if dx in malignant_classes else "benign"

df["binary_label"] = df["dx"].apply(label_binary)

# Train/Validation split (80/20)
train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["binary_label"],
    random_state=42
)

def copy_images(dataframe, split_name):
    for _, row in dataframe.iterrows():
        image_id = row["image_id"]
        label = row["binary_label"]
        filename = image_id + ".jpg"

        src = images_part1 / filename
        if not src.exists():
            src = images_part2 / filename

        if src.exists():
            dst = output_root / split_name / label / filename
            shutil.copy(src, dst)

# Copy files
print("Copying training images...")
copy_images(train_df, "train")

print("Copying validation images...")
copy_images(valid_df, "valid")

print("Done creating binary dataset.")
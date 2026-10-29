"""
check_data_split.py
Verify that the CIFAR-10 test set was truly held out and check data split sizes.
"""
import sys, pickle, numpy as np
from pathlib import Path

sys.path.insert(0, r"D:\FYP\ShiftGuard\results")

PROJECT_ROOT = Path(r"D:\FYP\ShiftGuard")
CIFAR_DATA_DIR = PROJECT_ROOT / "data" / "cifar-10-batches-py"

# Load all training batches
all_train_images, all_train_labels = [], []
for i in range(1, 6):
    with open(CIFAR_DATA_DIR / f"data_batch_{i}", "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    imgs = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    lbls = np.asarray(batch[b"labels"], dtype=np.int64)
    all_train_images.append(imgs)
    all_train_labels.append(lbls)
    print(f"Train batch {i}: {len(imgs):,} images")

train_images = np.concatenate(all_train_images, axis=0)
train_labels = np.concatenate(all_train_labels, axis=0)
print(f"\nTotal training: {len(train_images):,} images")

# Count automobile (1) and truck (9) in training set
auto_mask = train_labels == 1
truck_mask = train_labels == 9
print(f"Automobile in training: {auto_mask.sum():,}")
print(f"Truck in training: {truck_mask.sum():,}")
print(f"Auto+Truck in training: {(auto_mask | truck_mask).sum():,}")

# Load test batch
with open(CIFAR_DATA_DIR / "test_batch", "rb") as f:
    test_batch = pickle.load(f, encoding="bytes")
test_images = test_batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
test_labels = np.asarray(test_batch[b"labels"], dtype=np.int64)
print(f"\nTest set: {len(test_images):,} images")

auto_test = (test_labels == 1).sum()
truck_test = (test_labels == 9).sum()
print(f"Automobile in test: {auto_test:,}")
print(f"Truck in test: {truck_test:,}")
print(f"Auto+Truck in test: {auto_test + truck_test:,}")

# Verify no overlap between train and test sets
train_keys = set()
for i, lbl in zip(train_images, train_labels):
    # Use hash of image bytes as unique key
    train_keys.add((lbl, i.tobytes()))

overlap = 0
test_auto_truck_labels = test_labels[np.isin(test_labels, [1, 9])]
test_auto_truck_imgs = test_images[np.isin(test_labels, [1, 9])]
for lbl, img in zip(test_auto_truck_labels, test_auto_truck_imgs):
    if (lbl, img.tobytes()) in train_keys:
        overlap += 1

print(f"\nOverlap check (train vs test auto/truck): {overlap} duplicate images found (should be 0)")

# Verify the 90/10 split size
n_auto_truck = (auto_mask | truck_mask).sum()
n_val = max(1, int(0.10 * n_auto_truck))
n_train = n_auto_truck - n_val
print(f"\n90/10 split: {n_train} train, {n_val} validation (from {n_auto_truck} auto/truck pool)")

from pathlib import Path
import re
import json

ROOT = Path(__file__).resolve().parent

SKIP_DIRS = {
    ".git", ".venv", "venv", "env",
    "__pycache__", "node_modules"
}

PATTERNS = {
    "Optimizer": r"\b(AdamW?|SGD|RMSprop|optimizer|optimiser)\b",
    "Learning rate": r"\b(lr|learning_rate|base_lr)\b",
    "Batch size": r"\b(batch_size|batchsize)\b",
    "Epochs": r"\b(epochs?|num_epochs|total_epochs)\b",
    "Early stopping": r"\b(early.?stopping|patience)\b",
    "Loss function": r"\b(CrossEntropyLoss|criterion|loss_fn|loss_function)\b",
    "Augmentation": (
        r"\b(RandomHorizontalFlip|RandomCrop|RandomRotation|"
        r"ColorJitter|Resize|Normalize|augmentation|augmentations)\b"
    ),
    "Dataset split": (
        r"\b(train_test_split|random_split|train_size|val_size|"
        r"validation_split|val_dataset|train_dataset|seed)\b"
    ),
}

def main():
    print("PROJECT:", ROOT)

    extensions = {".py", ".json", ".txt", ".log", ".yaml", ".yml"}
    total = 0

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if any(part in SKIP_DIRS for part in path.parts):
            continue

        if path.suffix.lower() not in extensions:
            continue

        try:
            if path.stat().st_size > 5_000_000:
                continue

            content = path.read_text(
                encoding="utf-8", errors="ignore"
            )
        except OSError:
            continue

        matches = []

        for line_no, line in enumerate(content.splitlines(), 1):
            for category, pattern in PATTERNS.items():
                if re.search(pattern, line, re.IGNORECASE):
                    matches.append(
                        (line_no, category, line.strip())
                    )
                    break

        if matches:
            print(f"\nFILE: {path.relative_to(ROOT)}")

            for line_no, category, line in matches:
                print(f"  [{category}] Line {line_no}: {line[:200]}")
                total += 1

    print("\nTOTAL MATCHES:", total)

    print("\nCHECKPOINT FILES:")

    found = False

    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".pth", ".pt", ".ckpt"
        }:
            if any(part in SKIP_DIRS for part in path.parts):
                continue

            size = path.stat().st_size / (1024 * 1024)
            print(f"{path.relative_to(ROOT)} ({size:.2f} MB)")
            found = True

    if not found:
        print("No checkpoints found.")

    print("\nInspection completed.")

if __name__ == "__main__":
    main()
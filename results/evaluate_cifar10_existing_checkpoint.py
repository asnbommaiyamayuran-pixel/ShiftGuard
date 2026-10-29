import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RESULTS_DIR))

from shiftguard_pipeline import load_checkpoint  # noqa: E402

CHECKPOINT_PATH = PROJECT_ROOT / "shiftguard_checkpoint.pth"
CIFAR_TEST_PATH = PROJECT_ROOT / "data" / "cifar-10-batches-py" / "test_batch"

def load_cifar_batch(path):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels

def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not CIFAR_TEST_PATH.exists():
        raise FileNotFoundError(
            f"CIFAR-10 test_batch not found: {CIFAR_TEST_PATH}\n"
            "Expected extracted folder: data/cifar-10-batches-py/"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch, loss, stored_acc = load_checkpoint(
        filepath=str(CHECKPOINT_PATH), device=device
    )
    model.eval()

    images, labels = load_cifar_batch(CIFAR_TEST_PATH)
    # CIFAR-10 label 1 = automobile, label 9 = truck.
    selected = np.isin(labels, [1, 9])
    images = images[selected]
    labels = labels[selected]
    y_true = np.where(labels == 1, 0, 1)  # automobile->car, truck->truck

    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    class_names = ["car", "truck", "bike"]
    all_predictions = []
    all_confidences = []

    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start:start + batch_size]
            tensors = torch.stack([
                preprocess(Image.fromarray(img.astype(np.uint8)).convert("RGB"))
                for img in batch_imgs
            ]).to(device)

            output = model(tensors)
            # Handle models that return either logits or a tuple/dict.
            if isinstance(output, dict):
                logits = output.get("logits", output.get("class_logits"))
                if logits is None:
                    raise RuntimeError(f"Could not find logits in output keys: {list(output.keys())}")
            elif isinstance(output, (tuple, list)):
                logits = output[0]
            else:
                logits = output

            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            all_predictions.extend(pred.cpu().numpy().tolist())
            all_confidences.extend(conf.cpu().numpy().tolist())

    y_pred = np.asarray(all_predictions, dtype=np.int64)
    confidences = np.asarray(all_confidences, dtype=np.float64)

    # 2x3 matrix makes wrong 'bike' predictions visible rather than hiding them.
    cm = np.zeros((2, 3), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if 0 <= pred < 3:
            cm[truth, pred] += 1

    accuracy = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
    )

    print("\n=== ShiftGuard CIFAR-10 Cross-Domain Evaluation ===")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Device:", device)
    print(f"Checkpoint metadata only: epoch={epoch}, loss={loss:.4f}, stored_accuracy={stored_acc:.4f}")
    print("Evaluation scope: CIFAR-10 test images from automobile and truck classes only.")
    print(f"Images evaluated: {len(y_true)}")
    print(f"Ground-truth automobile images: {(y_true == 0).sum()}")
    print(f"Ground-truth truck images: {(y_true == 1).sum()}")
    print(f"Test accuracy (bike predictions count as wrong): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro precision (car/truck): {p:.4f}")
    print(f"Macro recall (car/truck):    {r:.4f}")
    print(f"Macro F1 (car/truck):        {f1:.4f}")
    print(f"Mean max-softmax confidence (descriptive only): {confidences.mean():.4f}")

    print("\nConfusion matrix: rows=true [automobile/car, truck], columns=predicted [car, truck, bike]")
    print("                  pred_car  pred_truck  pred_bike")
    print(f"true_automobile   {cm[0,0]:8d}  {cm[0,1]:10d}  {cm[0,2]:9d}")
    print(f"true_truck        {cm[1,0]:8d}  {cm[1,1]:10d}  {cm[1,2]:9d}")

    print("\nPer-class report (predictions of bike count as errors):")
    print(classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["automobile→car", "truck"],
        zero_division=0, digits=4
    ))

    print("\nIMPORTANT:")
    print("- This is a cross-domain test, not a standard full CIFAR-10 benchmark.")
    print("- The checkpoint was trained for three classes on synthetic road-scene data.")
    print("- Do not describe the checkpoint's stored 85% as this test accuracy.")
    print("- Save these results only after rerunning and keeping the console output.")

if __name__ == "__main__":
    main()

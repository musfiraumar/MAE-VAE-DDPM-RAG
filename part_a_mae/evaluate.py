import json
import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # repo root (MAE-VAE-DDPM-RAG)
sys.path.insert(0, str(ROOT))

from starter.eurosat import CLASS_NAMES   # <- fix to your starter filename

PRED_PATH = "part_a_mae/results/test_predictions.json"
CM_PATH = "part_a_mae/results/confusion_matrix.png"
RESULTS_PATH = "part_a_mae/results/classifier_results.json"

with open(PRED_PATH) as f:
    data = json.load(f)

test_pred = torch.tensor(data["pred"])
y_test = torch.tensor(data["label"])
train_acc = data["train_accuracy_160"]

test_acc = (test_pred == y_test).float().mean().item()
print(f"Test accuracy (5400 images): {test_acc * 100:.2f}%")

conf = torch.zeros(10, 10, dtype=torch.long)
for true, pred in zip(y_test.tolist(), test_pred.tolist()):
    conf[true, pred] += 1

print("Row sums:", conf.sum(dim=1).tolist())

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(conf.numpy(), cmap="Blues")

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (Test Accuracy: {test_acc * 100:.1f}%)")

thresh = conf.max().item() / 2
for i in range(10):
    for j in range(10):
        ax.text(j, i, int(conf[i, j]), ha="center", va="center", fontsize=8,
                color="white" if conf[i, j] > thresh else "black")

plt.colorbar(im)
plt.tight_layout()
plt.savefig(CM_PATH, dpi=150)
plt.show()

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "test_accuracy": test_acc,
        "train_accuracy_160": train_acc,
        "confusion_matrix": conf.tolist()
    }, f, indent=4)
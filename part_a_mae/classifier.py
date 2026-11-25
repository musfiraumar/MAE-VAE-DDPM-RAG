import json
import torch
import torch.nn as nn
from torch.optim import AdamW
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # repo root (MAE-VAE-DDPM-RAG)
sys.path.insert(0, str(ROOT))
from starter.eurosat import load_eurosat_split, few_label_indices, patchify   # <- fix to your starter filename
from model import MAE

CKPT_PATH = "part_a_mae/results/mae_final.pth"
PRED_PATH = "part_a_mae/results/test_predictions.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train, y_train = load_eurosat_split("train")
x_test, y_test = load_eurosat_split("test")
few_indices = few_label_indices(y_train)

train_patches = patchify(x_train)
test_patches = patchify(x_test)

model = MAE().to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print("Loaded weights from epoch", ckpt["epoch"])
model.eval()

@torch.no_grad()
def extract_features(encoder, patches, bs=512):
    encoder.eval()
    feats = []
    for i in range(0, len(patches), bs):
        out, _, _ = encoder(patches[i:i + bs].to(device), mask_on=False)
        feats.append(out.mean(dim=1).cpu())
    return torch.cat(feats)

train_feats = extract_features(model.encoder, train_patches[few_indices])
train_labels = y_train[few_indices]
test_feats = extract_features(model.encoder, test_patches)

print("Few-label features:", tuple(train_feats.shape))
print("Test features:", tuple(test_feats.shape))
print("Labels per class:", torch.bincount(train_labels).tolist())

train_feats = train_feats.to(device)
train_labels = train_labels.to(device)
test_feats = test_feats.to(device)
y_test_dev = y_test.to(device)

torch.manual_seed(0)

clf = nn.Linear(192, 10).to(device)
clf_opt = AdamW(clf.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(200):
    clf.train()
    clf_opt.zero_grad()
    loss = criterion(clf(train_feats), train_labels)
    loss.backward()
    clf_opt.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/200 - Loss: {loss.item():.4f}")

clf.eval()
with torch.no_grad():
    train_pred = clf(train_feats).argmax(dim=1)
    test_pred = clf(test_feats).argmax(dim=1)
    train_acc = (train_pred == train_labels).float().mean().item()
    test_acc = (test_pred == y_test_dev).float().mean().item()

print(f"Train accuracy (160 images): {train_acc * 100:.2f}%")
print(f"Test accuracy (5400 images): {test_acc * 100:.2f}%")

with open(PRED_PATH, "w") as f:
    json.dump({
        "pred": test_pred.cpu().tolist(),
        "label": y_test.tolist(),
        "train_accuracy_160": train_acc
    }, f)
print("Saved predictions to", PRED_PATH)
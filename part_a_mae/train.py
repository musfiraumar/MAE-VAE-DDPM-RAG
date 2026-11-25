import json
import time
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # repo root (MAE-VAE-DDPM-RAG)
sys.path.insert(0, str(ROOT))
from starter.eurosat import load_eurosat_split, patchify   
from model import MAE

EPOCHS = 100
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data (labels are not used in pretraining)
x_train, _ = load_eurosat_split("train")
train_patches = patchify(x_train)
train_loader = DataLoader(train_patches, batch_size=256, shuffle=True)

results_dir = Path("part_a_mae/results")
results_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = results_dir / "mae_final.pth"
loss_path = results_dir / "loss_history.json"

model = MAE().to(device)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

start_epoch = 0
loss_history = []

if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    if loss_path.exists():
        with open(loss_path, "r") as file:
            loss_history = json.load(file)

    print("Resuming from epoch:", start_epoch + 1)

else:
    print("Starting training from epoch 1")
parameter_count = sum(p.numel() for p in model.parameters())
print("Total parameters:", parameter_count)
start_time = time.time()

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0
    total_images = 0

    for patches in train_loader:
        patches = patches.to(device)
        optimizer.zero_grad()
        loss, predictions, mask = model(patches)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * patches.size(0)
        total_images += patches.size(0)
    epoch_loss = total_loss / total_images
    loss_history.append(epoch_loss)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {epoch_loss:.4f}")

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_loss
    }

    torch.save(checkpoint, checkpoint_path)

    if (epoch + 1) % 10 == 0:
        torch.save(
            checkpoint,
            results_dir / f"mae_epoch_{epoch + 1}.pth"
        )

    with open(loss_path, "w") as file:
        json.dump(loss_history, file, indent=4)

training_time = time.time() - start_time

with open(results_dir / "training_info.json", "w") as file:
    json.dump(
        {
            "epochs": len(loss_history),
            "parameters": parameter_count,
            "training_time_seconds": training_time,
            "final_loss": loss_history[-1]
        },
        file,
        indent=4
    )

print("Training completed.")
print("Total epochs:", len(loss_history))
print("Training time:", round(training_time / 60, 2), "minutes")
print("Final loss:", loss_history[-1])
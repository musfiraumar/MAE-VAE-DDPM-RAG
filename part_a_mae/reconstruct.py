import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent     
sys.path.insert(0, str(ROOT))
from starter.eurosat import load_eurosat_split, patchify, unpatchify  
from model import MAE

CKPT_PATH = "part_a_mae/results/mae_final.pth"
OUT_PATH = "part_a_mae/results/reconstructions.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_test, _ = load_eurosat_split("test")
test_patches = patchify(x_test)

model = MAE().to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print("Loaded weights from epoch", ckpt["epoch"])

model.eval()
torch.manual_seed(0)

idx = torch.randperm(len(test_patches))[:8]
original = test_patches[idx].to(device)

with torch.no_grad():
    _, pred, mask = model(original)

pred = pred.clamp(0, 1)

masked_input = original.clone()
masked_input[mask.bool()] = 0.5

reconstruction = torch.where(mask.bool().unsqueeze(-1), pred, original)

def to_imgs(patches):
    return unpatchify(patches).permute(0, 2, 3, 1).cpu().numpy()

rows = [to_imgs(masked_input), to_imgs(reconstruction), to_imgs(original)]
titles = ["Masked Input", "Reconstruction", "Original"]

fig, axes = plt.subplots(3, 8, figsize=(16, 6.5))
for i in range(3):
    for j in range(8):
        axes[i, j].imshow(rows[i][j].clip(0, 1))
        axes[i, j].axis("off")
    axes[i, 0].set_title(titles[i], loc="left", fontsize=11)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
plt.show()
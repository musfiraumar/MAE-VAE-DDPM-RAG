import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from diffusion import DEVICE, q_sample, sample


def forward_figure(images):
    x = images[:1].to(DEVICE)
    noise = torch.randn_like(x)

    fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))

    for ax, t in zip(axes, [0, 100, 250, 500, 750, 999]):
        xt = q_sample(x, torch.tensor([t], device=DEVICE), noise)
        ax.imshow(((xt + 1) / 2).clamp(0, 1)[0, 0].cpu(), cmap="gray")
        ax.set_title(f"t={t}")
        ax.axis("off")

    plt.savefig("results/forward_process.png", bbox_inches="tight")
    plt.show()


def grid_figure(unet, class_names):
    labels = torch.arange(40).repeat_interleave(8)
    images = sample(unet, labels).cpu()
    images = ((images + 1) / 2).clamp(0, 1)

    fig, axes = plt.subplots(40, 8, figsize=(8, 40))

    for i in range(40):
        for j in range(8):
            axes[i, j].imshow(images[i * 8 + j, 0], cmap="gray")
            axes[i, j].axis("off")

        axes[i, 0].set_ylabel(class_names[i], rotation=0, labelpad=30)

    plt.savefig("results/grid_40x8.png", bbox_inches="tight", dpi=100)
    plt.show()


def denoising_figure(unet, class_id=5):
    steps = [999, 750, 500, 250, 100, 0]
    _, saved = sample(unet, torch.tensor([class_id]), save_steps=steps[:-1])

    fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))

    for ax, t in zip(axes, steps):
        image = ((saved[t].clamp(-1, 1) + 1) / 2)[0, 0].cpu()
        ax.imshow(image, cmap="gray")
        ax.set_title(f"t={t}")
        ax.axis("off")

    plt.savefig("results/denoising.png", bbox_inches="tight")
    plt.show()


def make_eval_sheet(unet, class_names, seed=123):
    images = sample(unet, torch.arange(40)).cpu()
    images = ((images + 1) / 2).clamp(0, 1)

    perm = torch.randperm(40, generator=torch.Generator().manual_seed(seed))

    with open("results/KEY_do_not_open.json", "w") as f:
        json.dump({
            "true_labels": perm.tolist(),
            "class_names": list(class_names)
        }, f)

    fig, axes = plt.subplots(5, 8, figsize=(12, 8))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[perm[i], 0], cmap="gray")
        ax.set_title(f"#{i}")
        ax.axis("off")

    plt.savefig("results/human_eval_images.png", bbox_inches="tight")
    plt.show()


def score_eval(csv_path="results/human_eval.csv"):
    with open("results/KEY_do_not_open.json") as f:
        key = json.load(f)

    df = pd.read_csv(csv_path)
    df["true"] = [key["class_names"][i] for i in key["true_labels"]]

    acc1 = (df["member1"] == df["true"]).mean()
    acc2 = (df["member2"] == df["true"]).mean()
    agreement = (df["member1"] == df["member2"]).mean()

    print(f"Member 1: {acc1:.2%}")
    print(f"Member 2: {acc2:.2%}")
    print(f"Agreement: {agreement:.2%}")

    df.to_csv(csv_path, index=False)

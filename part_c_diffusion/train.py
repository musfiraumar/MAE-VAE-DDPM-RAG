import os, time
import torch
import matplotlib.pyplot as plt
from diffusion import DEVICE, T, q_sample

def train(unet, loader, epochs=40, lr=2e-4, ckpt_path="ddpm.pt"):
    os.makedirs("results", exist_ok=True)
    opt = torch.optim.Adam(unet.parameters(), lr=lr)
    loss_history, start, total_time = [], 0, 0.0

    if os.path.exists(ckpt_path):              # resume after Colab disconnect
        ck = torch.load(ckpt_path, map_location=DEVICE)
        unet.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        loss_history, start, total_time = ck["loss_history"], ck["epoch"], ck["total_time"]
        print("Resumed from epoch", start)

    for epoch in range(start, epochs):
        unet.train()
        t0, running = time.time(), 0.0
        for x0, y in loader:
            x0, y = x0.to(DEVICE), y.to(DEVICE)
            t = torch.randint(0, T, (x0.shape[0],), device=DEVICE)   # one t per image
            eps = torch.randn_like(x0)
            eps_hat = unet(q_sample(x0, t, eps), t, class_labels=y).sample
            loss = ((eps - eps_hat) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()

        loss_history.append(running / len(loader))
        total_time += time.time() - t0
        print(f"Epoch {epoch+1}/{epochs}  loss {loss_history[-1]:.4f}  time {total_time/60:.1f} min")
        torch.save({"model": unet.state_dict(), "opt": opt.state_dict(),
                    "loss_history": loss_history, "epoch": epoch + 1,
                    "total_time": total_time}, ckpt_path)

    plt.plot(loss_history); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.savefig("results/loss_curve.png", bbox_inches="tight"); plt.show()
    gpu = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    print("Final loss:", loss_history[-1], "| GPU:", gpu, "| minutes:", total_time / 60)
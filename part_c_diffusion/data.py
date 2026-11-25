import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_data(data_dir="/content/uhat_data/uhat", batch_size=128):
    d = np.load(f"{data_dir}/uhat_dataset.npz", allow_pickle=True)
    print("keys in npz:", d.files)

    # pick the train arrays (adjust here if your key names are different)
    x_key = next(k for k in d.files if "x" in k.lower() and "train" in k.lower())
    y_key = next(k for k in d.files if "y" in k.lower() and "train" in k.lower())
    images = torch.tensor(d[x_key]).float()
    labels = torch.tensor(d[y_key]).long()

    names_key = next((k for k in d.files if "name" in k.lower() or "class" in k.lower()), None)
    class_names = [str(n) for n in d[names_key]] if names_key else [str(i) for i in range(40)]

    if images.max() > 1.5:                 # pixels 0-255 -> 0-1
        images = images / 255.0
    if images.ndim == 3:                   # (N,28,28) -> (N,1,28,28)
        images = images.unsqueeze(1)
    images = images * 2 - 1                # [0,1] -> [-1,1]

    print("images:", images.shape, "range:", images.min().item(), images.max().item())
    print("labels:", labels.min().item(), "to", labels.max().item())

    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)
    return loader, images, class_names
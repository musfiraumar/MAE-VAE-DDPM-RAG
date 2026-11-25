import torch
from torch.utils.data import DataLoader
from starter.eurosat import (
    load_eurosat_split,
    few_label_indices,
    patchify,
    unpatchify
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load EuroSAT data
x_train, y_train = load_eurosat_split("train")
x_test, y_test = load_eurosat_split("test")

# Select 16 labelled images from each class
few_indices = few_label_indices(y_train)

# Convert images into patches
train_patches = patchify(x_train)
test_patches = patchify(x_test)

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)
print("Few labelled images:", len(few_indices))
print("Training patches:", train_patches.shape)
print("Testing patches:", test_patches.shape)

# DataLoader for MAE pretraining
train_loader = DataLoader(
    train_patches,
    batch_size=256,
    shuffle=True
)

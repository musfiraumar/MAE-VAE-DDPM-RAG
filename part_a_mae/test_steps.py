import torch
from part_a_mae.data import train_patches, few_indices
from part_a_mae.model import random_masking
# Test random masking
x = train_patches[:8]

visible, mask, ids_restore = random_masking(x)

print("Original patches:", x.shape)
print("Visible patches:", visible.shape)
print("Mask shape:", mask.shape)
print("Restore indices:", ids_restore.shape)
print("Masked patches per image:", mask.sum(dim=1))

assert visible.shape == (8, 16, 192)
assert mask.shape == (8, 64)
assert ids_restore.shape == (8, 64)
assert torch.all(mask.sum(dim=1) == 48)

print("All tests passed!")

import torch
from part_a_mae.model import MAE, MAEEncoder, MAEDecoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test input
x = torch.rand(8, 64, 192).to(device)
# Test encoder with masking
encoder = MAEEncoder().to(device)
encoded, mask, ids_restore = encoder(x)

print("Masked encoder output:", encoded.shape)
print("Mask shape:", mask.shape)

assert encoded.shape == (8, 16, 192)
assert mask.shape == (8, 64)
assert torch.all(mask.sum(dim=1) == 48)

# Test encoder without masking
encoded_full, _, _ = encoder(x, mask_on=False)
print("Full encoder output:", encoded_full.shape)
assert encoded_full.shape == (8, 64, 192)

# Test decoder
decoder = MAEDecoder().to(device)
predictions = decoder(encoded, ids_restore)
print("Decoder output:", predictions.shape)

assert predictions.shape == (8, 64, 192)

# Test complete MAE
model = MAE().to(device)

loss, predictions, mask = model(x)

print("MAE loss:", loss.item())
print("MAE predictions:", predictions.shape)
print("Total parameters:", sum(p.numel() for p in model.parameters()))

assert predictions.shape == (8, 64, 192)
assert mask.shape == (8, 64)
assert loss.ndim == 0

print("All model tests passed!")
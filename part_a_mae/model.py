import torch
import torch.nn as nn
def random_masking(x, keep=16):
    batch_size, num_patches, dim = x.shape

    noise = torch.rand(batch_size, num_patches, device=x.device)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :keep]

    visible = torch.gather(
        x,
        1,
        ids_keep.unsqueeze(-1).expand(-1, -1, dim)
    )

    mask = torch.ones(
        batch_size,
        num_patches,
        device=x.device
    )

    mask[:, :keep] = 0
    mask = torch.gather(mask, 1, ids_restore)

    return visible, mask, ids_restore


class MAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embed = nn.Linear(192, 192)

        self.pos_embed = nn.Parameter(
            torch.randn(1, 64, 192) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=3,
            dim_feedforward=768,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=6
        )

        self.norm = nn.LayerNorm(192)

    def forward(self, patches, mask_on=True):
        x = self.patch_embed(patches)
        x = x + self.pos_embed
        mask = None
        ids_restore = None
        if mask_on:
            x, mask, ids_restore = random_masking(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x, mask, ids_restore

class MAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.projection = nn.Linear(192, 128)

        self.mask_token = nn.Parameter(
            torch.randn(1, 1, 128) * 0.02
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, 64, 128) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=512,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=2
        )

        self.output = nn.Linear(128, 192)

    def forward(self, enc_tokens, ids_restore):
        x = self.projection(enc_tokens)
        batch_size = x.shape[0]
        mask_tokens = self.mask_token.expand(
            batch_size, 48, 128
        )
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x,
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, 128)
        )
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.output(x)

class MAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = MAEEncoder()
        self.decoder = MAEDecoder()

    def forward(self, patches):
        enc_tokens, mask, ids_restore = self.encoder(
            patches,
            mask_on=True
        )
        predictions = self.decoder(enc_tokens, ids_restore)
        loss_per_patch = (
            (predictions - patches) ** 2
        ).mean(dim=-1)
        loss = (loss_per_patch * mask).sum() / mask.sum()
        return loss, predictions, mask

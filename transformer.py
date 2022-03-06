import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    def __init__(self, n_hidden, n_heads, mlp_expansion, norm, lin_kqv) -> None:
        super().__init__()
        # Attention module
        self.attn = nn.MultiheadAttention(n_hidden, n_heads, batch_first=True)

        # TODO: Support normalizations
        if norm:
            raise NotImplementedError("Normalization not supported yet")

        # TODO: Support lin_kqv
        if lin_kqv:
            raise NotImplementedError(
                "Linear transformation for KQV not supported yet")

        # FF layer
        self.ff = nn.Sequential(
            nn.Linear(n_hidden, int(n_hidden*mlp_expansion)),
            nn.ReLU(),
            nn.Linear(int(n_hidden*mlp_expansion), n_hidden))

    def forward(self, x):
        # x shape: [B, T, C]

        # First pass through attention
        x = x + self.attn(x, x)

        # Then pass through FF
        return x + self.ff(x)


class TransformerBackbone(nn.Module):
    """
        Handles only one direction.
        Handles positional embedding.
    """

    def __init__(
                self,
                n_hidden=64,
                n_heads=4,
                n_layers=4,
                mlp_expansion=2,
                norm=False,
                lin_kqv=False,
                mask_perc=0.1,
            ) -> None:

        super().__init__()

        # Store info and get layers
        self.mask_perc = 0.1
        self.layers = nn.ModuleList([
            TransformerLayer(
                n_hidden, n_heads, mlp_expansion, norm, lin_kqv
            ) for _ in range(n_layers)
        ])

        # Positional encoding for injecting positional info
        # TODO
        self.pos_enc = None

        # Masked token
        self.masked_token = nn.Parameter(torch.randn(1, 1, n_hidden))

    def rand_mask(self, x):
        num_mask = int(x.shape[1]*self.mask_perc)

        # Select which to mask out
        # TODO: Use scatter so it's more efficient
        selected = torch.randperm(x.shape[1])[:num_mask]
        for idx in selected:
            x[:, idx, :] = self.masked_token
        return x

    def apply_pos_enc(self, x):
        return x+self.pos_enc

    def forward(self, x, mask=True):
        if mask:
            # First rand mask
            x = self.rand_mask(x)

        # Apply positional encoding
        x = self.apply_pos_enc(x)

        # Finally apply the transformer layers
        return self.layers(x)


class MultiViewTransformer(TransformerBackbone):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create direction embeddings
        self.dir_emb = nn.Parameter(torch.randn(1, 1, 3, kwargs['n_hidden']))

    def apply_dir_emb(self, x, n):
        return x[:, :, n]+self.dir_emb[:, :, n]

    def forward(self, x, mask=True):
        # x shape: [B, T, 3, C]
        outputs = []
        for dir_idx in range(3):
            outputs.append(
                super()(self.apply_dir_emb(x, dir_idx), mask).unsqueeze(2))

        return torch.cat(outputs, dim=2)


# TODO: A subclass of backbone that prediction

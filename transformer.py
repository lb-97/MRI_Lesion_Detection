import torch
import torch.nn as nn
import pytorch_lightning as pl


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
        # FIX Fix this 

        # FF layer
        self.ff = nn.Sequential(
            nn.Linear(n_hidden, int(n_hidden*mlp_expansion)),
            nn.ReLU(),
            nn.Linear(int(n_hidden*mlp_expansion), n_hidden))

    def forward(self, x):
        # x shape: [B, T, C]

        # First pass through attention
        x = x + self.attn(x, x, x)[0]

        # Then pass through FF
        return x + self.ff(x)


class TransformerBackbone(nn.Module):
    """
        Handles only one direction.
        Handles positional embedding.
    """

    def __init__(
                self,
                n_hidden=10,
                n_heads=2,
                n_layers=4,
                mlp_expansion=2,
                norm=False,
                lin_kqv=False,
                mask_perc=0.1,
                max_length=600
            ) -> None:

        super().__init__()

        # Store info and get layers
        self.mask_perc = mask_perc
        self.layers = nn.Sequential(*[
            TransformerLayer(
                n_hidden, n_heads, mlp_expansion, norm, lin_kqv
            ) for _ in range(n_layers)
        ])

        # Positional encoding for injecting positional info
        self.pos_enc = nn.Parameter(
            torch.randn(1, max_length, n_hidden)
        )
        self.n_hidden = n_hidden

        # Masked token
        self.masked_token = nn.Parameter(torch.randn(1, n_hidden))

    def rand_mask(self, x):
        num_mask = int(x.shape[1]*self.mask_perc)

        # Select which to mask out
        selected = torch.randperm(x.shape[1])[:num_mask]
        for idx in selected:
            x[:, idx, :] = self.masked_token
        return x

    def apply_pos_enc(self, x):
        return x+self.pos_enc[:,:x.shape[1]]

    def forward(self, x, mask=False):
        if mask:
            # First rand mask
            x = self.rand_mask(x)

        # Apply positional encoding
        x = self.apply_pos_enc(x)

        # Finally apply the transformer layers
        return self.layers(x)


class MultiViewTransformer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # Create direction embeddings
        n_hidden = self.hparams.n_hidden
        self.dir_emb = nn.Parameter(torch.randn(1, 1, 3, n_hidden))
        self.backbone = TransformerBackbone(n_hidden=args.n_hidden)

    def apply_dir_emb(self, x):
        return x+self.dir_emb

    def forward(self, x, mask=False):
        B, T, _, C = x.shape
        input_reshaped = self.apply_dir_emb(x).flatten(1, 2)
        return self.backbone(input_reshaped, mask).reshape(B, T, 3, C)

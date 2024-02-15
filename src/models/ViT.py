import numpy as np
import torch
import torch.nn as nn


class PatchExtractor(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images):
        batch, _, height, width = images.size()
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input height ({height}) and width ({width}) must be divisible by self.patch_size ({self.patch_size})"

        h_patches = height // self.patch_size
        w_patches = width // self.patch_size

        n_patches = h_patches * w_patches

        patches = (
            images.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(batch, n_patches, -1)
        )

        return patches


class InputEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.training.device
        self.latent_dim = cfg.model.latent_dim
        self.batch_size = cfg.training.batch_size
        self.patch_size = cfg.model.patch_size
        self.in_dim = self.patch_size * self.patch_size * 3
        self.linear_projection = nn.Linear(
            self.in_dim, self.latent_dim, device=self.device
        )
        self.class_tokens = nn.Parameter(
            torch.randn((self.batch_size, 1, self.latent_dim), device=self.device)
        )

    def _get_pos_enc(self, seq_len, d, n=10000):
        pe = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in range(d // 2):
                denominator = n ** (2 * i / d)
                pe[k, 2 * i] = np.sin(k / denominator)
                pe[k, 2 * i + 1] = np.cos(k / denominator)

        return torch.from_numpy(np.expand_dims(pe, axis=0))

    def forward(self, x):
        patchify = PatchExtractor(patch_size=self.patch_size)
        patches = patchify(x)

        embedding = self.linear_projection(patches)
        seq_len = embedding.shape[1]
        pos_enc = self._get_pos_enc(seq_len + 1, self.latent_dim).to(self.device)
        embedding = torch.cat((embedding, self.class_tokens), dim=1)
        embedding += pos_enc
        return embedding


class Encoder(nn.Module):
    def __init__(self, train_cfg):
        super().__init__()
        self.device = train_cfg.training.device
        self.latent_dim = train_cfg.model.latent_dim
        self.num_heads = train_cfg.model.num_heads
        self.dropout = train_cfg.training.dropout

        self.layer_norm = nn.LayerNorm(self.latent_dim, device=self.device)
        self.attention_block = nn.MultiheadAttention(
            self.latent_dim, self.num_heads, dropout=self.dropout, device=self.device
        )
        self.mlp_block = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim * 4, self.latent_dim),
            nn.Dropout(self.dropout),
        )
        self.mlp_block.to(self.device)

    def forward(self, x):
        # Pre-layer Normalization
        norm_x = self.layer_norm(x)
        attn_out = self.attention_block(norm_x, norm_x, norm_x)[0]  # Self-Attention
        first_add = attn_out + x
        norm_attn = self.layer_norm(first_add)
        mlp_out = self.mlp_block(norm_attn)
        out = mlp_out + first_add
        return out


class ViT(nn.Module):
    def __init__(self, data_cfg, train_cfg):
        super().__init__()
        self.num_encoders = train_cfg.model.num_encoders
        self.num_classes = data_cfg.num_classes
        self.latent_dim = train_cfg.model.latent_dim
        self.device = train_cfg.training.device

        self.embedding = InputEmbeddings(train_cfg)
        self.encoders = nn.Sequential(
            *[Encoder(train_cfg) for i in range(self.num_encoders)]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.num_classes),
        )
        self.mlp_head.to(self.device)

    def forward(self, x):
        x_embedding = self.embedding(x)
        encoded_class_token = self.encoders(x_embedding)[:, 0, :]
        logits = self.mlp_head(encoded_class_token)
        return logits

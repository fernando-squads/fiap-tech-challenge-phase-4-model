from __future__ import annotations

import torch
from torch import nn


class MultimodalTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dims: dict[str, int],
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims nao pode ser vazio.")
        if d_model % nhead != 0:
            raise ValueError("d_model deve ser divisivel por nhead.")

        self.modalities = list(input_dims.keys())
        self.projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(input_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for modality, input_dim in input_dims.items()
            }
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.modality_embeddings = nn.Parameter(torch.zeros(len(self.modalities) + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.modality_embeddings, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        first_modality = self.modalities[0]
        batch_size = batch[first_modality].shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = [cls + self.modality_embeddings[0].view(1, 1, -1)]
        padding_mask = [
            torch.zeros(batch_size, dtype=torch.bool, device=cls.device)
        ]

        for index, modality in enumerate(self.modalities, start=1):
            projected = self.projections[modality](batch[modality])
            projected = projected + self.modality_embeddings[index].view(1, -1)
            tokens.append(projected.unsqueeze(1))
            present = batch[f"{modality}_present"].to(dtype=torch.bool, device=projected.device)
            padding_mask.append(~present)

        token_tensor = torch.cat(tokens, dim=1)
        mask_tensor = torch.stack(padding_mask, dim=1)
        encoded = self.encoder(token_tensor, src_key_padding_mask=mask_tensor)
        logits = self.classifier(encoded[:, 0, :]).squeeze(-1)
        return logits


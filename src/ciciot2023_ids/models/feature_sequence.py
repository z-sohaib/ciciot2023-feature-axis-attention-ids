from __future__ import annotations

import torch
from torch import nn


class FeatureSequenceMultiScaleBiLSTMAttention(nn.Module):
    """Feature-axis multi-scale CNN-BiLSTM-attention classifier.

    This is the architecture family selected on UNSW-NB15. It treats each
    encoded feature position as a sequence step so convolution, BiLSTM, and
    attention operate across feature positions.
    """

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        conv_channels: int,
        kernel_sizes: tuple[int, ...],
        lstm_hidden: int,
        lstm_layers: int,
        attention_heads: int,
        dense_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("At least one convolution kernel size is required.")

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=conv_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(conv_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for kernel_size in kernel_sizes
            ]
        )
        cnn_features = conv_channels * len(kernel_sizes)
        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        attention_dim = lstm_hidden * 2
        if attention_dim % attention_heads != 0:
            raise ValueError("BiLSTM output dimension must be divisible by attention_heads.")
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        h = torch.cat(branch_outputs, dim=1)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        attended, _ = self.attention(h, h, h, need_weights=False)
        pooled = attended.mean(dim=1)
        return self.classifier(pooled)


class FeatureSequenceMultiScaleBiGRUAttention(nn.Module):
    """Lighter feature-axis multi-scale CNN-BiGRU-attention classifier."""

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        conv_channels: int,
        kernel_sizes: tuple[int, ...],
        gru_hidden: int,
        gru_layers: int,
        attention_heads: int,
        dense_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("At least one convolution kernel size is required.")

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=conv_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(conv_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for kernel_size in kernel_sizes
            ]
        )
        cnn_features = conv_channels * len(kernel_sizes)
        self.gru = nn.GRU(
            input_size=cnn_features,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        attention_dim = gru_hidden * 2
        if attention_dim % attention_heads != 0:
            raise ValueError("BiGRU output dimension must be divisible by attention_heads.")
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        h = torch.cat(branch_outputs, dim=1)
        h = h.transpose(1, 2)
        h, _ = self.gru(h)
        attended, _ = self.attention(h, h, h, need_weights=False)
        pooled = attended.mean(dim=1)
        return self.classifier(pooled)


class FTTransformerClassifier(nn.Module):
    """FT-Transformer-style classifier for row-level tabular features."""

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        transformer_dim: int,
        transformer_layers: int,
        attention_heads: int,
        dense_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if transformer_dim % attention_heads != 0:
            raise ValueError("transformer_dim must be divisible by attention_heads.")

        self.value_embedding = nn.Linear(1, transformer_dim)
        self.feature_embedding = nn.Parameter(torch.zeros(1, input_features, transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=attention_heads,
            dim_feedforward=dense_hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.transpose(1, 2).unsqueeze(-1)
        tokens = self.value_embedding(values.squeeze(2)) + self.feature_embedding
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        h = torch.cat([cls, tokens], dim=1)
        h = self.encoder(h)
        return self.classifier(h[:, 0])


class TabTransformerStyleClassifier(nn.Module):
    """TabTransformer-style neural baseline for continuous CICIoT2023 features."""

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        transformer_dim: int,
        transformer_layers: int,
        attention_heads: int,
        dense_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if transformer_dim % attention_heads != 0:
            raise ValueError("transformer_dim must be divisible by attention_heads.")

        self.value_embedding = nn.Sequential(
            nn.Linear(1, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
        )
        self.feature_embedding = nn.Parameter(torch.zeros(1, input_features, transformer_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=attention_heads,
            dim_feedforward=dense_hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.transpose(1, 2).unsqueeze(-1)
        tokens = self.value_embedding(values.squeeze(2)) + self.feature_embedding
        h = self.encoder(tokens)
        return self.classifier(h.mean(dim=1))


class TabNetLikeClassifier(nn.Module):
    """Lightweight attentive TabNet-style baseline without extra dependencies."""

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        dense_hidden: int,
        decision_steps: int,
        relaxation_factor: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.relaxation_factor = relaxation_factor
        self.attention_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_features, dense_hidden),
                    nn.BatchNorm1d(dense_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dense_hidden, input_features),
                )
                for _ in range(decision_steps)
            ]
        )
        self.feature_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_features, dense_hidden),
                    nn.BatchNorm1d(dense_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dense_hidden, dense_hidden),
                    nn.BatchNorm1d(dense_hidden),
                    nn.GELU(),
                )
                for _ in range(decision_steps)
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(dense_hidden),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x.squeeze(1)
        prior = torch.ones_like(features)
        aggregated: torch.Tensor | None = None
        for attention_layer, feature_block in zip(self.attention_layers, self.feature_blocks):
            mask_logits = attention_layer(features)
            mask = torch.softmax(mask_logits * prior, dim=1)
            transformed = feature_block(features * mask)
            aggregated = transformed if aggregated is None else aggregated + transformed
            prior = torch.clamp(prior * (self.relaxation_factor - mask), min=0.0)
        if aggregated is None:
            raise RuntimeError("TabNetLikeClassifier requires at least one decision step.")
        return self.classifier(aggregated)

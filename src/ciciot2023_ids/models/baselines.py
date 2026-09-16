from __future__ import annotations

import torch
from torch import nn


class TabularMLP(nn.Module):
    """Plain DNN/MLP baseline for row-level CICIoT2023 features."""

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        hidden_layers: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_features
        for hidden in hidden_layers:
            layers.extend(
                [
                    nn.Linear(width, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            width = hidden
        layers.append(nn.Linear(width, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.squeeze(1))


class FeatureCNN(nn.Module):
    def __init__(self, input_features: int, num_classes: int, conv_channels: int, dense_hidden: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


class FeatureLSTM(nn.Module):
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        lstm_hidden: int,
        lstm_layers: int,
        dense_hidden: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        width = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h, _ = self.lstm(h)
        return self.classifier(h.mean(dim=1))


class FeatureCNNBiLSTM(nn.Module):
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        conv_channels: int,
        lstm_hidden: int,
        lstm_layers: int,
        dense_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.classifier(h.mean(dim=1))

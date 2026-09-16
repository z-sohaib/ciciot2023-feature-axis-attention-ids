from __future__ import annotations

from torch import nn

from ciciot2023_ids.models.baselines import FeatureCNN, FeatureCNNBiLSTM, FeatureLSTM, TabularMLP
from ciciot2023_ids.models.feature_sequence import (
    FTTransformerClassifier,
    FeatureSequenceMultiScaleBiGRUAttention,
    FeatureSequenceMultiScaleBiLSTMAttention,
    TabNetLikeClassifier,
    TabTransformerStyleClassifier,
)


def build_model(config: dict, input_features: int, num_classes: int) -> nn.Module:
    model_config = config["model"]
    architecture = model_config["architecture"]
    if architecture == "cnn":
        return FeatureCNN(
            input_features=input_features,
            num_classes=num_classes,
            conv_channels=int(model_config["conv_channels"]),
            dense_hidden=int(model_config["dense_hidden"]),
            dropout=float(model_config["dropout"]),
        )
    if architecture == "mlp":
        hidden_layers = model_config.get("hidden_layers", [256, 128])
        return TabularMLP(
            input_features=input_features,
            num_classes=num_classes,
            hidden_layers=tuple(int(value) for value in hidden_layers),
            dropout=float(model_config["dropout"]),
        )
    if architecture == "lstm":
        return FeatureLSTM(
            input_features=input_features,
            num_classes=num_classes,
            lstm_hidden=int(model_config["lstm_hidden"]),
            lstm_layers=int(model_config["lstm_layers"]),
            dense_hidden=int(model_config["dense_hidden"]),
            dropout=float(model_config["dropout"]),
            bidirectional=False,
        )
    if architecture == "cnn_bilstm":
        return FeatureCNNBiLSTM(
            input_features=input_features,
            num_classes=num_classes,
            conv_channels=int(model_config["conv_channels"]),
            lstm_hidden=int(model_config["lstm_hidden"]),
            lstm_layers=int(model_config["lstm_layers"]),
            dense_hidden=int(model_config["dense_hidden"]),
            dropout=float(model_config["dropout"]),
        )
    if architecture == "feature_attention":
        return FeatureSequenceMultiScaleBiLSTMAttention(
            input_features=input_features,
            num_classes=num_classes,
            conv_channels=int(model_config["conv_channels"]),
            kernel_sizes=tuple(int(value) for value in model_config["kernel_sizes"]),
            lstm_hidden=int(model_config["lstm_hidden"]),
            lstm_layers=int(model_config["lstm_layers"]),
            attention_heads=int(model_config["attention_heads"]),
            dense_hidden=int(model_config["dense_hidden"]),
            dropout=float(model_config["dropout"]),
        )
    if architecture == "feature_gru_attention":
        return FeatureSequenceMultiScaleBiGRUAttention(
            input_features=input_features,
            num_classes=num_classes,
            conv_channels=int(model_config["conv_channels"]),
            kernel_sizes=tuple(int(value) for value in model_config["kernel_sizes"]),
            gru_hidden=int(model_config.get("gru_hidden", model_config.get("lstm_hidden", 96))),
            gru_layers=int(model_config.get("gru_layers", model_config.get("lstm_layers", 1))),
            attention_heads=int(model_config["attention_heads"]),
            dense_hidden=int(model_config["dense_hidden"]),
            dropout=float(model_config["dropout"]),
        )
    if architecture == "ft_transformer":
        return FTTransformerClassifier(
            input_features=input_features,
            num_classes=num_classes,
            transformer_dim=int(model_config.get("transformer_dim", model_config.get("dense_hidden", 128))),
            transformer_layers=int(model_config.get("transformer_layers", 2)),
            attention_heads=int(model_config["attention_heads"]),
            dense_hidden=int(model_config["dense_hidden"]),
            dropout=float(model_config["dropout"]),
        )
    if architecture == "tab_transformer":
        return TabTransformerStyleClassifier(
            input_features=input_features,
            num_classes=num_classes,
            transformer_dim=int(model_config.get("transformer_dim", model_config.get("dense_hidden", 128))),
            transformer_layers=int(model_config.get("transformer_layers", 2)),
            attention_heads=int(model_config["attention_heads"]),
            dense_hidden=int(model_config["dense_hidden"]),
            dropout=float(model_config["dropout"]),
        )
    if architecture == "tabnet_like":
        return TabNetLikeClassifier(
            input_features=input_features,
            num_classes=num_classes,
            dense_hidden=int(model_config["dense_hidden"]),
            decision_steps=int(model_config.get("decision_steps", 4)),
            relaxation_factor=float(model_config.get("relaxation_factor", 1.5)),
            dropout=float(model_config["dropout"]),
        )
    raise ValueError(f"Unknown architecture: {architecture}")

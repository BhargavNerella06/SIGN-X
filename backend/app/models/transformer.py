"""Temporal Transformer for isolated sign recognition.

Pipeline:

    Input Projection -> Positional Encoding -> Transformer Encoder
    -> Temporal (attention) Pooling -> Dropout -> Classification Head

Input:  (batch, sequence_length, feature_dim)
Output: (batch, num_classes) logits

Every architectural knob (embedding dim, layers, heads, dropout, feed
forward size, max sequence length) is a constructor argument — nothing is
hardcoded, so ``training/config.yaml`` and ``app/config.py`` fully control
the model shape.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, embedding_dim: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class AttentionPooling(nn.Module):
    """Learned attention pooling over the time dimension: instead of a
    plain mean, the model learns which frames matter most for the sign
    (e.g. the peak of the gesture rather than transitional frames)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attn = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, seq_len, embedding_dim)
        scores = self.attn(x).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        return (x * weights).sum(dim=1)  # (batch, embedding_dim)


class SignTransformer(nn.Module):
    """Temporal Transformer encoder for sequence classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.2,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pooling = AttentionPooling(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: optional (batch, seq_len) bool tensor, True = real frame,
                  False = padding. Used both to prevent the encoder from
                  attending to padded positions and to exclude them from
                  pooling.
        Returns:
            logits: (batch, num_classes)
        """
        h = self.input_projection(x)
        h = self.positional_encoding(h)

        key_padding_mask = ~mask if mask is not None else None  # True = ignore
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        pooled = self.pooling(h, mask=mask)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

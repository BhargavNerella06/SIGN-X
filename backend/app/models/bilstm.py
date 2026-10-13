"""BiLSTM baseline classifier — used to benchmark the Transformer against a
simpler, well-established recurrent architecture on the same feature
sequences.

Input:  (batch, sequence_length, feature_dim)
Output: (batch, num_classes) logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SignBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: optional (batch, seq_len) bool tensor, True = real frame.
                  Used to compute each sequence's true (unpadded) length so
                  the final hidden state reflects the last real frame, not
                  a padding frame.
        """
        h = self.input_norm(x)

        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths, batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(h)

        # hidden: (num_layers * 2, batch, hidden_dim) -> take last layer's
        # forward + backward states and concatenate.
        forward_last = hidden[-2]
        backward_last = hidden[-1]
        combined = torch.cat([forward_last, backward_last], dim=-1)

        combined = self.dropout(combined)
        return self.classifier(combined)

"""Multi-kernel 1D Convolutional Neural Network (TextCNN) for Email Classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmailTextCNN(nn.Module):
    """
    1D CNN architecture using parallel n-gram convolutional filters (2, 3, 4-grams),
    global max-over-time pooling, dropout regularization, and a linear classification head.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (2, 3, 4),
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len) -> embedded: (batch_size, embed_dim, seq_len)
        embedded = self.embedding(x).transpose(1, 2)
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)
            pooled_outputs.append(pooled)
        cat = torch.cat(pooled_outputs, dim=1)
        return self.fc(self.dropout(cat))

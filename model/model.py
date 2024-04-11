import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, base, quantiles):
        super().__init__()
        self.base_model = base
        self.output = nn.Linear(len(quantiles), 1)

    def freeze(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        quantile_output = self.base_model(x)
        return torch.cat([quantile_output, self.output(quantile_output)], dim=-1)

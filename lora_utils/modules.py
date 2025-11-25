import torch
import torch.nn as nn
import math


class LoRA(nn.Module):
    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base_layer = base_layer
        self.scales = []
        self.dropouts = nn.ModuleList() # Dropout層を保持
        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def init_lora(self,rank,alpha,dropout=0.0):
        self.scales.append(alpha / rank if rank > 0 else 1.0)
        self.dropouts.append(nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())

        if isinstance(self.base_layer, nn.Linear):
            self.lora_A.append(nn.Linear(self.base_layer.in_features, rank, bias=False))
            self.lora_B.append(nn.Linear(rank, self.base_layer.out_features, bias=False))
        elif isinstance(self.base_layer, nn.Conv2d):
            self.lora_A.append(nn.Conv2d(self.base_layer.in_channels, rank, kernel_size=1, bias=False))
            self.lora_B.append(
                nn.Conv2d(rank, self.base_layer.out_channels, kernel_size=1, stride=self.base_layer.stride, padding=self.base_layer.padding, bias=False)
            )

        nn.init.kaiming_uniform_(self.lora_A[-1].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[-1].weight)


    def load_weight(self, lora_A, lora_B, strength=1.0, alpha=1.0, dropout=0.0, idx=None):
        idx = -1 if idx is None else idx
        rank = lora_A.shape[0]
        alpha = alpha*strength
        self.init_lora(rank,alpha,dropout)
        self.lora_A[idx].weight.data.copy_(lora_A)
        self.lora_B[idx].weight.data.copy_(lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.base_layer(x)

        for i in range(len(self.lora_A)):
            a_module, b_module = self.lora_A[i], self.lora_B[i]
            scale = self.scales[i]
            dropout = self.dropouts[i]
            w += scale * dropout(b_module(a_module(x)))
        
        return w
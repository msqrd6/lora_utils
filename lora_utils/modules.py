import torch
import torch.nn as nn
import math

class BufferList(nn.Module):
    """
    nn.ParameterListのように振る舞うが、中身をBuffer(学習対象外)として管理するクラス。
    state_dict上では 'alpha.0', 'alpha.1' のようなキーになります。
    """
    def __init__(self):
        super().__init__()
        
    def append(self, tensor):
        # 現在の要素数を名前にして登録 ("0", "1", "2"...)
        # これにより 親.alpha.0 という名前が生成される
        name = str(len(self._buffers))
        self.register_buffer(name, tensor)
        
    def __getitem__(self, idx):
        # alpha[i] でアクセス可能にする
        if idx < 0:
            idx = len(self._buffers) + idx
        return getattr(self, str(idx))

    def __len__(self):
        return len(self._buffers)

class LoRA(nn.Module):
    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base_layer = base_layer
        self.scales = []
        self.dropouts = nn.ModuleList() # Dropout層を保持
        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()
        self.alpha = BufferList()

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def append_lora_layer(self,rank,alpha,strength=1.0,dropout=0.0):
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        self.scales.append(strength * (alpha / rank) if rank > 0 else 1.0)
        self.dropouts.append(nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())
        
        alpha_tensor = alpha.detach().clone().float() if isinstance(alpha,torch.Tensor) else torch.tensor(alpha, dtype=torch.float32)
        alpha_tensor = alpha_tensor.to(device=device,dtype=dtype)
        #self.alpha.append(nn.Parameter(alpha_tensor,requires_grad=False))
        self.alpha.append(alpha_tensor.to(device=device,dtype=dtype))


        if isinstance(self.base_layer, nn.Linear):
            a = nn.Linear(self.base_layer.in_features, rank, bias=False)
            b = nn.Linear(rank, self.base_layer.out_features, bias=False)
        elif isinstance(self.base_layer, nn.Conv2d):
            a = nn.Conv2d(self.base_layer.in_channels, rank, kernel_size=1, bias=False)
            b = nn.Conv2d(rank, self.base_layer.out_channels, kernel_size=1, stride=self.base_layer.stride, padding=self.base_layer.padding, bias=False)
        else:
            return
        
        self.lora_A.append(a.to(device=device,dtype=dtype))
        self.lora_B.append(b.to(device=device,dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_A[-1].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[-1].weight)

    def load_weight(self, lora_A, lora_B, strength=1.0, alpha=1.0, dropout=0.0):
        rank = lora_A.shape[0]
        self.append_lora_layer(rank,alpha,strength,dropout)
        self.alpha[-1].data.fill_(float(alpha))
        self.lora_A[-1].weight.data.copy_(lora_A)
        self.lora_B[-1].weight.data.copy_(lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.base_layer(x)

        for i in range(len(self.lora_A)):
            a_module, b_module = self.lora_A[i], self.lora_B[i]
            scale = self.scales[i]
            dropout = self.dropouts[i]
            w += scale * dropout(b_module(a_module(x)))
        
        return w
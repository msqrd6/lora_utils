import torch
import torch.nn as nn
from copy import deepcopy
import math

try:
    from optimum.quanto.nn import QLinear, QConv2d
    QUANTO_AVAILABLE = True
except ImportError:
    QUANTO_AVAILABLE = False
    # quantoがない環境でもエラーにならないようにダミークラスを作成
    class QLinear: pass
    class QConv2d: pass

class BufferList(nn.Module):
    def __init__(self):
        super().__init__()
        
    def append(self, tensor):
        name = str(len(self._buffers))
        self.register_buffer(name, tensor)
        
    def __getitem__(self, idx):
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
        self.dropouts = nn.ModuleList() 
        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()
        self.alpha = BufferList()

        # base_layerのパラメータを凍結
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def append_lora_layer(self, rank, alpha, strength=1.0, dropout=0.0):
        device = self.base_layer.weight.device
        
        # ==========================================
        # ★修正: dtypeの取得ロジック (重要)
        # ==========================================
        # 量子化されている場合、元のweightはint8だが、LoRAはfloatで計算する必要がある
        dtype = self.base_layer.weight.dtype
        if dtype in [torch.int8, torch.uint8, torch.qint8]:
            # A100環境ならbfloat16推奨。汎用性を取るならfloat32
            dtype = torch.bfloat16 
        
        self.scales.append(strength * (alpha / rank) if rank > 0 else 1.0)
        self.dropouts.append(nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())
        
        alpha_tensor = alpha.detach().clone().float() if isinstance(alpha, torch.Tensor) else torch.tensor(alpha, dtype=torch.float32)
        alpha_tensor = alpha_tensor.to(device=device, dtype=dtype)
        self.alpha.append(alpha_tensor)

        # ==========================================
        # ★修正: QLinear / QConv2d の判定を追加
        # ==========================================
        if isinstance(self.base_layer, (nn.Linear, QLinear)):
            in_features = self.base_layer.in_features
            out_features = self.base_layer.out_features
            
            a = nn.Linear(in_features, rank, bias=False)
            b = nn.Linear(rank, out_features, bias=False)
            
        elif isinstance(self.base_layer, (nn.Conv2d, QConv2d)):
            in_channels = self.base_layer.in_channels
            out_channels = self.base_layer.out_channels
            stride = self.base_layer.stride
            padding = self.base_layer.padding
            
            a = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
            b = nn.Conv2d(rank, out_channels, kernel_size=1, stride=stride, padding=padding, bias=False)
        else:
            print(f"Skipping unsupported layer type: {type(self.base_layer)}")
            return
        
        self.lora_A.append(a.to(device=device, dtype=dtype))
        self.lora_B.append(b.to(device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.lora_A[-1].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[-1].weight)

    def load_weight(self, lora_A, lora_B, strength=1.0, alpha=1.0, dropout=0.0):
        rank = lora_A.shape[0]
        self.append_lora_layer(rank, alpha, strength, dropout)
        self.alpha[-1].data.fill_(float(alpha))
        self.lora_A[-1].weight.data.copy_(lora_A)
        self.lora_B[-1].weight.data.copy_(lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # quantoのレイヤーはforwardで自動的にdequantizeして計算してくれるのでそのまま呼べばOK
        w = self.base_layer(x)

        for i in range(len(self.lora_A)):
            a_module, b_module = self.lora_A[i], self.lora_B[i]
            scale = self.scales[i]
            dropout = self.dropouts[i]
            
            # LoRA計算
            delta_w = b_module(a_module(x))
            w += scale * dropout(delta_w)
        
        return w

def _inject_empty_lora_layer(model, module_name):
    parent_module = model
    path = module_name.split(".")
    for p in path[:-1]:
        parent_module = getattr(parent_module, p)
    last_name = path[-1]
    base_layer = getattr(parent_module, last_name)

    if isinstance(base_layer, LoRA):
        return base_layer
     
    lora_layer = LoRA(base_layer)
    setattr(parent_module, last_name, lora_layer)
    return lora_layer

def inject_lora(model, rank=4, alpha=1.0, dropout=0.0, inject_layer_key:list[str]=[], linear:bool=True, conv2d:bool=True):
    def needs_lora_injection(module_name):
        if len(inject_layer_key) == 0:
            return True
        for key in inject_layer_key:
            if key in module_name:
                return True
        return False
    
    target_modules = []

    for module_name, module in model.named_modules():
        if needs_lora_injection(module_name):
            # ==========================================
            # ★修正: QLinear / QConv2d の判定を追加
            # ==========================================
            if linear:
                if isinstance(module, (nn.Linear, QLinear)):
                    target_modules.append((module_name, module))
            if conv2d:
                if isinstance(module, (nn.Conv2d, QConv2d)):
                    target_modules.append((module_name, module))

    for module_name, module in target_modules:
        # LoRAレイヤーが既に注入されているかチェック
        # (再帰的に走査していると親と子が重複する場合があるため)
        if isinstance(module, LoRA):
            continue
            
        lora_layer = _inject_empty_lora_layer(model, module_name)
        lora_layer.append_lora_layer(rank, alpha, strength=1.0, dropout=dropout)


def load_lora(base_model, lora_state_dict, strength=1.0, lora_requires_grad=False):
    for key, value in lora_state_dict.items():
        if not "lora_A" in key: continue
        base_key = key.split(".lora_A.")[0]
        lora_A = value
        lora_B = lora_state_dict.get(base_key + '.lora_B.weight')
        rank = lora_A.shape[0]
        alpha = lora_state_dict.get(base_key + '.alpha', rank/2)
        
        lora_layer = _inject_empty_lora_layer(base_model, base_key)
        lora_layer.load_weight(lora_A, lora_B, strength, alpha)
    
    if not lora_requires_grad:
        base_model.requires_grad_(False)

    return base_model


def unload_lora(model):

    for name, child in model.named_children():
        # もし子モジュールが LoRA クラスなら
        if isinstance(child, LoRA):
            # 1. 元の層 (base_layer) を取り出す
            original_layer = child.base_layer
            if isinstance(model, nn.Sequential) or isinstance(model, nn.ModuleList):
                model[int(name)] = original_layer
            else:
                setattr(model, name, original_layer)
        else:
            unload_lora(child)

    return model


def marge_lora_and_weight(lora_state_dict,base_state_dict,strength=1.0):
    output_state_dict = deepcopy(base_state_dict)
    for key, value in lora_state_dict.items():
        if not "lora_A" in key: continue
        base_key = key.split(".lora_A.")[0]
        lora_A = value
        lora_B = lora_state_dict.get(base_key + '.lora_B.weight')
        rank = lora_A.shape[0]
        # .alphaが存在しない場合はrank/2を代入
        alpha = lora_state_dict.get(base_key + '.alpha', rank/2)
        scale = strength*alpha/rank
        if lora_A.dim() == 4:
            delta_W = (lora_B.squeeze() @ lora_A.squeeze())
            delta_W = delta_W.unsqueeze(-1).unsqueeze(-1)
        else:
            delta_W = (lora_B @ lora_A)

        with torch.no_grad():
            output_state_dict[base_key+".weight"] += scale*delta_W
    return output_state_dict


def get_lora_state_dict(model:nn.Module,get_model_dict=False):
    lora_state_dict = {}
    model_state_dict = {}

    with torch.no_grad():
        for key, value in model.state_dict().items():
            if "lora" in key:
                out_key = key.replace("0.weight","weight")
                lora_state_dict[out_key] = value
            elif "alpha" in key:
                out_key = key.replace("alpha.0","alpha")
                lora_state_dict[out_key] = value
            elif get_model_dict:
                if "base_layer.weight" in key:
                    out_key = key.replace("base_layer.weight","weight")
                elif "base_layer.bias" in key:
                    out_key = key.replace("base_layer.bias","bias")
                else:
                    out_key = key
                model_state_dict[out_key] = value

    if get_model_dict:
        return lora_state_dict, model_state_dict
    
    return lora_state_dict
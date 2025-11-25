import torch
import torch.nn as nn
from copy import deepcopy
from .modules import LoRA


def inject_init_lora_for_model(model, rank=4, alpha=1.0, dropout=0.0,inject_layer_key:list[str]=[],only_linear:bool=False):
    network_alphas = {}

   #loraを注入する層か判定
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
            if isinstance(module,(nn.Linear,nn.Conv2d)) and not only_linear:
                target_modules.append((module_name,module))
            elif isinstance(module,nn.Linear):
                target_modules.append((module_name,module))
    
    for module_name, module in target_modules:
            network_alphas[module_name+".alpha"] = torch.tensor(alpha)
            lora_layer = inject_empty_lora_layer(model,module_name)
            lora_layer.init_lora(rank,alpha,dropout)

    model = model.to(device=model.device,dtype=model.dtype)

    return network_alphas
    

def get_module_by_key(model, key):
    parts = key.split('.')
    module = model
    for p in parts[:-1]:
        if p.isdigit():
            module = module[int(p)]
        else:
            module = getattr(module, p)
    return module, parts[-1]


def inject_empty_lora_layer(model,module_name):
    parent_module = model
    path = module_name.split(".")
    for p in path[:-1]:
        parent_module = getattr(parent_module,p)
    last_name = path[-1]
    base_layer = getattr(parent_module, last_name)
    lora_layer = LoRA(base_layer)
    setattr(parent_module, last_name, lora_layer)
    return lora_layer



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


def separate_lora_from_model(model:nn.Module,out_model_state_dict=False):
    lora_state_dict = {}
    model_state_dict = {}

    with torch.no_grad():
        for key, value in model.state_dict().items():
            if "lora" in key:
                out_key = key.replace("0.weight","weight")
                lora_state_dict[out_key] = value
            elif out_model_state_dict:
                if "base_layer.weight" in key:
                    out_key = key.replace("base_layer.weight","weight")
                elif "base_layer.bias" in key:
                    out_key = key.replace("base_layer.bias","bias")
                else:
                    out_key = key
                model_state_dict[out_key] = value

    if not out_model_state_dict:
        return lora_state_dict
    
    return lora_state_dict, model_state_dict


def inject_pretrained_lora_for_model(base_model,lora_state_dict,strength=1.0):
    for key, value in lora_state_dict.items():
        if not "lora_A" in key: continue
        base_key = key.split(".lora_A.")[0]
        lora_A = value
        lora_B = lora_state_dict.get(base_key + '.lora_B.weight')
        rank = lora_A.shape[0]
        # .alphaが存在しない場合はrank/2を代入
        alpha = lora_state_dict.get(base_key + '.alpha', rank/2)
        
        lora_layer = inject_empty_lora_layer(base_model,base_key)
        lora_layer.load_weight(lora_A,lora_B,strength,alpha)
    base_model = base_model.to(device=base_model.device,dtype=base_model.dtype)


from lora_utils import inject_init_lora_for_model,inject_pretrained_lora_for_model, get_lora_dict_from_model,remove_lora_from_model
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(10,10)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(10,10)

    def forward(self,x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2
        return x

model = Model()

inject_init_lora_for_model(model,128,64)
sd = get_lora_dict_from_model(model)
remove_lora_from_model(model)



inject_pretrained_lora_for_model(model,sd)
inject_pretrained_lora_for_model(model,sd)







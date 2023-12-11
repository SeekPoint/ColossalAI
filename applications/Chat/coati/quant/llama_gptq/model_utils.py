# copied from https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/past/modelutils.py

import torch.nn as nn
from pydebug import gd, infoTensor

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    gd.debuginfo(prj="mt", info=f'')
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res

import torch.nn.functional as F
from pydebug import gd, infoTensor
from ...registry import bias_addition_function
from .bias_addition_function import LinearBasedBiasFunc


@bias_addition_function.register(F.linear)
class Linear(LinearBasedBiasFunc):
    def extract_kwargs_from_origin_func(self):
        gd.debuginfo(prj="mt", info=f'')
        assert "bias" in self.kwargs
        kwargs = {}
        if "bias" in self.kwargs:
            kwargs["bias"] = self.kwargs["bias"]
            gd.debuginfo(prj="mt", info=f'')
        return kwargs

    def generate(self):
        gd.debuginfo(prj="mt", info=f'')
        non_bias_linear_func_proxy = self.create_non_bias_func_proxy(self.args[0], self.args[1])
        kwargs = self.extract_kwargs_from_origin_func()
        bias_addition_proxy = self.create_bias_addition_proxy(non_bias_linear_func_proxy, kwargs["bias"])

        return bias_addition_proxy

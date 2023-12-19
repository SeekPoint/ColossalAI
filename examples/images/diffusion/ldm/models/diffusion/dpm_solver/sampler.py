"""SAMPLING ONLY."""
import torch

from .dpm_solver import DPM_Solver, NoiseScheduleVP, model_wrapper

MODEL_TYPES = {"eps": "noise", "v": "v"}
from pydebug import gd, infoTensor

class DPMSolverSampler(object):
    def __init__(self, model, **kwargs):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer("alphas_cumprod", to_torch(model.alphas_cumprod))
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def register_buffer(self, name, attr):
        gd.debuginfo(prj="mt", info=f'')
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    gd.debuginfo(prj="mt", info=f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    gd.debuginfo(prj="mt", info=f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        gd.debuginfo(prj="mt", info=f"Data shape for DPM-Solver sampling is {size}, sampling steps {S}")

        device = self.model.betas.device
        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T

        ns = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            ns,
            model_type=MODEL_TYPES[self.model.parameterization],
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        dpm_solver = DPM_Solver(model_fn, ns, predict_x0=True, thresholding=False)
        x = dpm_solver.sample(
            img, steps=S, skip_type="time_uniform", method="multistep", order=2, lower_order_final=True
        )
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        return x.to(device), None

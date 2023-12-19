import torch
from packaging.version import Version
from pydebug import gd, infoTensor
if Version(torch.__version__) >= Version("2.0.0"):
    from torch.optim.lr_scheduler import LRScheduler as _LRScheduler
else:
    from torch.optim.lr_scheduler import _LRScheduler


class _enable_get_lr_call:
    def __init__(self, o):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        self.o = o
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def __enter__(self):
        gd.debuginfo(prj="mt", info=f'')
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        gd.debuginfo(prj="mt", info=f'')
        self.o._get_lr_called_within_step = False


class DelayerScheduler(_LRScheduler):
    """Starts with a flat lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau)

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        delay_epochs (int): Number of epochs to keep the initial lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, delay_epochs, after_scheduler, last_epoch=-1):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        if delay_epochs < 0:
            raise ValueError(f"delay_epochs must >= 0, got {delay_epochs}")
        self.delay_epochs = delay_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in "optimizer"}
        gd.debuginfo(prj="mt", info=f'')
        if isinstance(state_dict["after_scheduler"], _LRScheduler):
            state_dict["after_scheduler_type"] = type(state_dict["after_scheduler"]).__name__
            state_dict["after_scheduler_dict"] = state_dict["after_scheduler"].state_dict()
            del state_dict["after_scheduler"]
            gd.debuginfo(prj="mt", info=f'')
        else:
            raise NotImplementedError()
        return state_dict

    def get_lr(self):
        gd.debuginfo(prj="mt", info=f'')
        if self.last_epoch >= self.delay_epochs:
            gd.debuginfo(prj="mt", info=f'')
            if not self.finished:
                gd.debuginfo(prj="mt", info=f'')
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            with _enable_get_lr_call(self.after_scheduler):
                return self.after_scheduler.get_lr()

        return self.base_lrs

    def step(self, epoch=None):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        if self.finished:
            if epoch is None:
                gd.debuginfo(prj="mt", info=f'')
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                gd.debuginfo(prj="mt", info=f'')
                self.after_scheduler.step(epoch - self.delay_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            gd.debuginfo(prj="mt", info=f'')
            return super(DelayerScheduler, self).step(epoch)

        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')


class WarmupScheduler(_LRScheduler):
    """Starts with a linear warmup lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to linearly warmup lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        self.warmup_epochs = int(warmup_epochs)
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def state_dict(self):
        gd.debuginfo(prj="mt", info=f'')
        state_dict = {key: value for key, value in self.__dict__.items() if key not in "optimizer"}
        if isinstance(state_dict["after_scheduler"], _LRScheduler):
            gd.debuginfo(prj="mt", info=f'')
            state_dict["after_scheduler_type"] = type(state_dict["after_scheduler"]).__name__
            state_dict["after_scheduler_dict"] = state_dict["after_scheduler"].state_dict()
            del state_dict["after_scheduler"]
        else:
            raise NotImplementedError()
        return state_dict

    def get_lr(self):
        gd.debuginfo(prj="mt", info=f'')
        if self.last_epoch >= self.warmup_epochs:
            gd.debuginfo(prj="mt", info=f'')
            if not self.finished:
                gd.debuginfo(prj="mt", info=f'')
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        if self.finished:
            if epoch is None:
                gd.debuginfo(prj="mt", info=f'')
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                gd.debuginfo(prj="mt", info=f'')
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            gd.debuginfo(prj="mt", info=f'')
            return super().step(epoch)
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')


class WarmupDelayerScheduler(_LRScheduler):
    """Starts with a linear warmup lr schedule until it reaches N epochs and a flat lr schedule
    until it reaches M epochs then applies the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to linearly warmup lr until starting applying the scheduler.
        delay_epochs (int): Number of epochs to keep the initial lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, warmup_epochs, delay_epochs, after_scheduler, last_epoch=-1):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        if delay_epochs < 0:
            raise ValueError(f"delay_epochs must >= 0, got {delay_epochs}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must >= 0, got {warmup_epochs}")
        self.warmup_epochs = warmup_epochs
        self.delay_epochs = delay_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def state_dict(self):
        gd.debuginfo(prj="mt", info=f'')
        state_dict = {key: value for key, value in self.__dict__.items() if key not in "optimizer"}
        if isinstance(state_dict["after_scheduler"], _LRScheduler):
            gd.debuginfo(prj="mt", info=f'')
            state_dict["after_scheduler_type"] = type(state_dict["after_scheduler"]).__name__
            state_dict["after_scheduler_dict"] = state_dict["after_scheduler"].state_dict()
            del state_dict["after_scheduler"]
        else:
            raise NotImplementedError()
        return state_dict

    def get_lr(self):
        gd.debuginfo(prj="mt", info=f'')
        if self.last_epoch >= self.warmup_epochs + self.delay_epochs:
            gd.debuginfo(prj="mt", info=f'')
            if not self.finished:
                gd.debuginfo(prj="mt", info=f'')
                self.after_scheduler.base_lrs = self.base_lrs
                # reset lr to base_lr
                for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    group["lr"] = base_lr
                self.finished = True
            with _enable_get_lr_call(self.after_scheduler):
                return self.after_scheduler.get_lr()
        elif self.last_epoch >= self.warmup_epochs:
            gd.debuginfo(prj="mt", info=f'')
            return self.base_lrs

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        if self.finished:
            if epoch is None:
                gd.debuginfo(prj="mt", info=f'')
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                gd.debuginfo(prj="mt", info=f'')
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            gd.debuginfo(prj="mt", info=f'')
            return super().step(epoch)

        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

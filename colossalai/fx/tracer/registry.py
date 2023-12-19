from pydebug import gd, infoTensor

class PatchRegistry:
    def __init__(self, name):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        self.name = name
        self.store = {}
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def register(self, source):
        gd.debuginfo(prj="mt", info=f'')
        def wrapper(func):
            gd.debuginfo(prj="mt", info=f'')
            self.store[source] = func
            return func

        return wrapper

    def get(self, source):
        assert source in self.store
        target = self.store[source]
        return target

    def has(self, source):
        return source in self.store


meta_patched_function = PatchRegistry(name="patched_functions_for_meta_execution")
meta_patched_module = PatchRegistry(name="patched_modules_for_meta_execution")
bias_addition_function = PatchRegistry(name="patched_function_for_bias_addition")
bias_addition_module = PatchRegistry(name="patched_module_for_bias_addition")
bias_addition_method = PatchRegistry(name="patched_method_for_bias_addition")

from pydebug import gd, infoTensor

class ProfilerRegistry:
    def __init__(self, name):
        self.name = name
        self.store = {}
        gd.debuginfo(prj="mt", info=f'')

    def register(self, source):
        def wrapper(func):
            self.store[source] = func
            return func

        gd.debuginfo(prj="mt", info=f'')
        return wrapper

    def get(self, source):
        gd.debuginfo(prj="mt", info=f'')
        assert source in self.store
        target = self.store[source]
        return target

    def has(self, source):
        gd.debuginfo(prj="mt", info=f'')
        return source in self.store


meta_profiler_function = ProfilerRegistry(name="patched_functions_for_meta_profile")
meta_profiler_module = ProfilerRegistry(name="patched_modules_for_meta_profile")

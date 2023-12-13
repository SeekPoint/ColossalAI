from pydebug import gd, infoTensor

class Registry:
    def __init__(self, name):
        gd.debuginfo(prj="mt", info=f'')
        self.name = name
        self.store = {}

    def register(self, source):
        gd.debuginfo(prj="mt", info=f'')
        def wrapper(func):
            if isinstance(source, (list, tuple)):
                # support register a list of items for this func
                for element in source:
                    self.store[element] = func
            else:
                self.store[source] = func
            return func

        return wrapper

    def get(self, source):
        assert source in self.store, f"{source} not found in the {self.name} registry"
        target = self.store[source]
        return target

    def has(self, source):
        return source in self.store


operator_registry = Registry("operator")

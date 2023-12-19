from pydebug import gd, infoTensor

class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        gd.debuginfo(prj="mt", info=f'cls.__name__={cls.__name__}')
        if cls not in cls._instances:
            gd.debuginfo(prj="mt", info=f'-----------0---------------')
            instance = super().__call__(*args, **kwargs)
            gd.debuginfo(prj="mt", info=f'instance={instance}')
            cls._instances[cls] = instance
        else:
            assert (
                len(args) == 0 and len(kwargs) == 0
            ), f"{cls.__name__} is a singleton class and a instance has been created."
        return cls._instances[cls]

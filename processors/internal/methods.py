from typing import Union, Callable


class ModuleInspector:
    @classmethod
    def get_methods(cls, obj : Union[object, type],
                    include_inherited: bool = True,
                    include_private = False,
                    include_magic_methods : bool = False) -> list[Callable]:

        def mthd_ok(mthd_name : str) -> bool:
            if cls.is_magical(mthd_name):
                return include_magic_methods
            elif cls.is_private(mthd_name):
                return include_private
            else:
                return True

        attrs = dir(obj)
        if not include_inherited:
            obj_cls = obj if isinstance(obj, type) else obj.__class__
            parent_attrs = []
            for p in obj_cls.__bases__:
                parent_attrs += dir(p)
            attrs = [name for name in attrs if not name in parent_attrs]
        attr_values = [getattr(obj, name) for name in attrs]

        targeted_methods = []
        for attr_name, value in zip(attrs, attr_values):
            if callable(value) and mthd_ok(attr_name):
                targeted_methods.append(value)

        return targeted_methods

    @staticmethod
    def is_magical(name : str):
        return name.startswith('__') and name.endswith('__')

    @staticmethod
    def is_private(name : str):
        return name.startswith('_')
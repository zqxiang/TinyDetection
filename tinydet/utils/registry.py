from typing import Optional
from typing import Dict

from tabulate import tabulate

__all__ = ["Registry"]


class Registry(object):
    def __init__(self, name: str) -> None:
        self._name = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object) -> None:
        assert name not in self._obj_map, f"An object named '{name}' was already registered in '{self._name}' registry!"

        self._obj_map[name] = obj
        
    def register(self, name: Optional[str] = None, obj: Optional[object] = None) -> Optional[object]:
        if obj is None:

            def deco(func_or_class: object) -> object:
                if name is None:
                    self._do_register(func_or_class.__name__, func_or_class)
                else:
                    self._do_register(name, func_or_class)
                return func_or_class

            return deco

        if name is None:
            name = obj.__name__
        self._do_register(name, obj)

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        table_header = ["Names", "Objects"]
        table = tabulate(self._obj_map.items(), headers=table_header, tablefmt="fancy_grid")
        return f"Registry of {self._name}\n" + table

    __str__ = __repr__

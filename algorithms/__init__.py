import importlib
import pkgutil

_EXCLUDED_MODULES = {"algorithm_interface", "registry", "symbolic"}

for _, module_name, _ in pkgutil.iter_modules(__path__):
    if module_name not in _EXCLUDED_MODULES:
        importlib.import_module(f"{__name__}.{module_name}")

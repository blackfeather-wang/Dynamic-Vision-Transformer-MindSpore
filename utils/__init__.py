"""Some utility unctions."""

from importlib import import_module

import mindspore.common.dtype as mstype

from .cfg_parser import ConfigObject


def get_ms_type(dtype):
    """Convert string type name to MindSpore type."""
    if dtype == "float32":
        return mstype.float32
    if dtype == "float16":
        return mstype.float16
    raise Exception("Unknown type.")

def load_function(func_name):
    """Load function using its name."""
    modules = func_name.split(".")
    if len(modules) > 1:
        module_path = ".".join(modules[:-1])
        name = modules[-1]

        module = import_module(module_path)
        return getattr(module, name)
    return func_name

def dynamic_call(args, inject_args=None, turn_on_func=False):
    """Call dynamically function or instantiate object using its name and args."""

    inject = inject_args if inject_args else {}

    if isinstance(args, ConfigObject):
        args = args.__dict__

    if isinstance(args, dict):
        args = dict(args)
        func = None
        if "func" in args:
            func = args.pop("func")
            func = load_function(func)
        elif ("_func" in args) and turn_on_func:
            func = args.pop("_func")
            func = load_function(func)

        for key, val in args.items():
            if val:
                args[key] = dynamic_call(val, inject)
            else:
                if key in inject:
                    args[key] = inject[key]

        if func and ("instant" in args):
            del args["instant"]

        return func(**args) if func else args
    if isinstance(args, list):
        objects = []
        for val in args:
            objects.append(dynamic_call(val, inject))
        return objects
    return args

def dump_net(net, filename):
    """Dump network structure to file."""
    with open(filename, "w") as file:
        for name, cell in net.cells_and_names():
            file.write(f'{name}: {type(cell).__name__}\n')

import pickle
import sys
from pathlib import Path

import numpy
import numpy.core.multiarray
import numpy.core.numeric


MODULE_ALIASES = {
    "numpy._core": numpy.core,
    "numpy._core.numeric": numpy.core.numeric,
    "numpy._core.multiarray": numpy.core.multiarray,
}


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in MODULE_ALIASES:
            sys.modules.setdefault(module, MODULE_ALIASES[module])
        if module == "numpy._core":
            module = "numpy.core"
        elif module == "numpy._core.numeric":
            module = "numpy.core.numeric"
        elif module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        return super().find_class(module, name)


def load_pickle_file(path):
    with Path(path).open("rb") as file_obj:
        return CompatUnpickler(file_obj, encoding="latin1").load()


def dump_pickle_file(path, payload):
    with Path(path).open("wb") as file_obj:
        pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)

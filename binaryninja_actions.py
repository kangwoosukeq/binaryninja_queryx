import binaryninja
import json

from dataclasses import dataclass, asdict

from .binaryninja_ast import Expr, parse_ast
from .constants import *


@dataclass
class Function:
    name: str
    ast: Expr


def dump(bv: binaryninja.BinaryView):
    functions = {}
    for func in bv.functions:
        ast = parse_ast(func.hlil.root)
        functions[func.start] = asdict(Function(func.name, ast))

    result = {}
    result[FUNCTIONS] = functions

    dump_path = binaryninja.get_save_filename_input(
        "Select a location for saved dump file"
    )

    if dump_path:
        with open(dump_path, "w") as f:
            f.write(json.dumps(result))

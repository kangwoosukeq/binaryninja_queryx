import binaryninja
import json

from dataclasses import dataclass, asdict

from .binaryninja_ast import Expr, parse_ast, Variable, parse_variable
from .constants import *


@dataclass
class Function:
    name: str
    vars: list[Variable]
    ast: Expr


def dump(bv: binaryninja.BinaryView):
    functions = {}
    for func in bv.functions:
        vars = {}
        for v in func.hlil.vars + func.hlil.aliased_vars:
            id = v.identifier
            if id in vars:
                continue
            vars[id] = parse_variable(v)

        ast = parse_ast(func.hlil.root)
        functions[func.start] = asdict(Function(func.name, vars, ast))

    result = {}
    result[ARCHITECTURE] = bv.arch.name
    result[FUNCTIONS] = functions

    dump_path = binaryninja.get_save_filename_input(
        "Select a location for the dump file"
    )

    if dump_path:
        with open(dump_path, "w") as f:
            f.write(json.dumps(result))

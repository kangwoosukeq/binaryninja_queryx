import binaryninja
import json
import traceback

from dataclasses import dataclass, asdict
from functools import partial

from .binaryninja_ast import Expr, parse_ast, Variable, parse_variable
from .binaryninja_ast import iter, VarDeclare, VarInit, Var
from .constants import *


@dataclass
class Function:
    name: str
    vars: list[Variable]
    ast: Expr


def log_info(text: str):
    binaryninja.log.log_info(text, QUERY_X)


def log_error(text: str):
    binaryninja.log.log_error(text, QUERY_X)


def verify_var(valid_var_ids: set[int], expr: Expr):
    if isinstance(expr, VarDeclare):
        assert expr.var in valid_var_ids
    elif isinstance(expr, VarInit):
        assert expr.dest in valid_var_ids
    elif isinstance(expr, Var):
        assert expr.var in valid_var_ids


def dump(bv: binaryninja.BinaryView):
    functions = {}
    for func in bv.functions:
        log_info(f"Processing for {func.name}...")

        vars = {}
        for v in func.hlil.vars + func.hlil.aliased_vars:
            id = v.identifier
            if id in vars:
                continue
            vars[id] = parse_variable(v)

        ast = parse_ast(func.hlil.root)
        log_info("Parsing AST complete...")

        try:
            iter(partial(verify_var, vars.keys()), ast)
        except AssertionError:
            log_error(f"Error occuring while verification!\n" + traceback.format_exc())
            return

        log_info("Verifying AST complete!")

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

import binaryninja

from binaryninja import HighLevelILOperation
from dataclasses import dataclass, fields

from .constants import *


@dataclass
class Expr:
    op: binaryninja.HighLevelILOperation
    address: int
    instr_index: binaryninja.InstructionIndex
    expr_index: binaryninja.ExpressionIndex


@dataclass
class Block(Expr):
    body: list[Expr]


@dataclass
class If(Expr):
    condition: Expr
    true: Expr
    false: Expr


@dataclass
class While(Expr):
    condition: Expr
    body: Expr


@dataclass
class DoWhile(Expr):
    body: Expr
    condition: Expr


@dataclass
class For(Expr):
    init: Expr
    condition: Expr
    update: Expr
    body: Expr


@dataclass
class Switch(Expr):
    condition: Expr
    default: Expr
    cases: list[Expr]


@dataclass
class Case(Expr):
    values: list[Expr]
    body: Expr


@dataclass
class Jump(Expr):
    dest: Expr


@dataclass
class Ret(Expr):
    src: list[Expr]


@dataclass
class GotoLabel:
    label_id: int
    name: str


def parse_goto_label(label: binaryninja.GotoLabel) -> GotoLabel:
    return GotoLabel(label.label_id, label.name)


@dataclass
class Goto(Expr):
    target: GotoLabel


@dataclass
class Label(Expr):
    target: GotoLabel


@dataclass
class Variable:
    index: int
    is_parameter: bool
    name: str
    source_type: binaryninja.VariableSourceType
    storage: int
    type: str | None


def parse_variable(var: binaryninja.Variable) -> Variable:
    return Variable(
        var.index,
        var.is_parameter_variable,
        var.name,
        var.source_type,
        var.storage,
        var.type.get_string() if var.type != None else None,
    )


@dataclass
class VarDeclare(Expr):
    var: int


@dataclass
class VarInit(Expr):
    dest: int
    src: Expr


@dataclass
class Assign(Expr):
    dest: Expr
    src: Expr


@dataclass
class AssignUnpack(Expr):
    dest: list[Expr]
    src: Expr


@dataclass
class Var(Expr):
    var: int


@dataclass
class StructField(Expr):
    src: Expr
    offset: int
    member_index: int | None


@dataclass
class ArrayIndex(Expr):
    src: Expr
    index: Expr


@dataclass
class Split(Expr):
    high: Expr
    low: Expr


@dataclass
class Unary(Expr):
    src: Expr


def parse_unary(expr: binaryninja.HighLevelILInstruction, args) -> Unary:
    src = parse_ast(expr.src)
    return Unary(*args, src)


@dataclass
class DerefField(Expr):
    src: Expr
    offset: int
    member_index: int | None


@dataclass
class Const(Expr):
    constant: int


@dataclass
class ConstData(Expr):
    constant_data: str


@dataclass
class ConstPtr(Expr):
    constant: int


@dataclass
class ExternPtr(Expr):
    constant: int
    offset: int


@dataclass
class FloatConst(Expr):
    constant: float


@dataclass
class Import(Expr):
    constant: int


@dataclass
class Binary(Expr):
    left: Expr
    right: Expr


def parse_binary(expr: binaryninja.HighLevelILInstruction, args) -> Binary:
    left = parse_ast(expr.left)
    right = parse_ast(expr.right)
    return Binary(*args, left, right)


@dataclass
class BinaryWithCarry(Binary):
    carry: Expr


def parse_binary_with_carry(
    expr: binaryninja.HighLevelILInstruction, args
) -> BinaryWithCarry:
    left = parse_ast(expr.left)
    right = parse_ast(expr.right)
    carry = parse_ast(expr.carry)
    return BinaryWithCarry(*args, left, right, carry)


@dataclass
class Call(Expr):
    dest: Expr
    params: list[Expr]


@dataclass
class Syscall(Expr):
    params: list[Expr]


@dataclass
class Tailcall(Expr):
    dest: Expr
    params: list[Expr]


@dataclass
class ILIntrinsic:
    index: binaryninja.IntrinsicIndex
    name: binaryninja.IntrinsicName


def parse_intrinsic(intrinsic: binaryninja.ILIntrinsic) -> ILIntrinsic:
    return ILIntrinsic(intrinsic.index, intrinsic.name)


@dataclass
class Intrinsic(Expr):
    intrinsic: ILIntrinsic
    params: list[Expr]


@dataclass
class Trap(Expr):
    vector: int


# Since we target HLIR, we didn't implemented class for *_SSA and *_PHI instructions.

unary_op_set = set(
    [
        HighLevelILOperation.HLIL_DEREF,
        HighLevelILOperation.HLIL_ADDRESS_OF,
        HighLevelILOperation.HLIL_NEG,
        HighLevelILOperation.HLIL_NOT,
        HighLevelILOperation.HLIL_SX,
        HighLevelILOperation.HLIL_ZX,
        HighLevelILOperation.HLIL_LOW_PART,
        HighLevelILOperation.HLIL_BOOL_TO_INT,
        HighLevelILOperation.HLIL_UNIMPL_MEM,
        HighLevelILOperation.HLIL_FSQRT,
        HighLevelILOperation.HLIL_FNEG,
        HighLevelILOperation.HLIL_FABS,
        HighLevelILOperation.HLIL_FLOAT_TO_INT,
        HighLevelILOperation.HLIL_INT_TO_FLOAT,
        HighLevelILOperation.HLIL_FLOAT_CONV,
        HighLevelILOperation.HLIL_ROUND_TO_INT,
        HighLevelILOperation.HLIL_FLOOR,
        HighLevelILOperation.HLIL_CEIL,
        HighLevelILOperation.HLIL_FTRUNC,
    ]
)

binary_op_set = set(
    [
        HighLevelILOperation.HLIL_ADD,
        HighLevelILOperation.HLIL_SUB,
        HighLevelILOperation.HLIL_AND,
        HighLevelILOperation.HLIL_OR,
        HighLevelILOperation.HLIL_XOR,
        HighLevelILOperation.HLIL_LSL,
        HighLevelILOperation.HLIL_LSR,
        HighLevelILOperation.HLIL_ASR,
        HighLevelILOperation.HLIL_ROL,
        HighLevelILOperation.HLIL_ROR,
        HighLevelILOperation.HLIL_MUL,
        HighLevelILOperation.HLIL_MULU_DP,
        HighLevelILOperation.HLIL_MULS_DP,
        HighLevelILOperation.HLIL_DIVU,
        HighLevelILOperation.HLIL_DIVU_DP,
        HighLevelILOperation.HLIL_DIVS,
        HighLevelILOperation.HLIL_DIVS_DP,
        HighLevelILOperation.HLIL_MODU,
        HighLevelILOperation.HLIL_MODU_DP,
        HighLevelILOperation.HLIL_MODS,
        HighLevelILOperation.HLIL_MODS_DP,
        HighLevelILOperation.HLIL_CMP_E,
        HighLevelILOperation.HLIL_CMP_NE,
        HighLevelILOperation.HLIL_CMP_SLT,
        HighLevelILOperation.HLIL_CMP_ULT,
        HighLevelILOperation.HLIL_CMP_SLE,
        HighLevelILOperation.HLIL_CMP_ULE,
        HighLevelILOperation.HLIL_CMP_SGE,
        HighLevelILOperation.HLIL_CMP_UGE,
        HighLevelILOperation.HLIL_CMP_SGT,
        HighLevelILOperation.HLIL_CMP_UGT,
        HighLevelILOperation.HLIL_TEST_BIT,
        HighLevelILOperation.HLIL_ADD_OVERFLOW,
        HighLevelILOperation.HLIL_FADD,
        HighLevelILOperation.HLIL_FSUB,
        HighLevelILOperation.HLIL_FMUL,
        HighLevelILOperation.HLIL_FDIV,
        HighLevelILOperation.HLIL_FCMP_E,
        HighLevelILOperation.HLIL_FCMP_NE,
        HighLevelILOperation.HLIL_FCMP_LT,
        HighLevelILOperation.HLIL_FCMP_LE,
        HighLevelILOperation.HLIL_FCMP_GE,
        HighLevelILOperation.HLIL_FCMP_GT,
        HighLevelILOperation.HLIL_FCMP_O,
        HighLevelILOperation.HLIL_FCMP_UO,
    ]
)

binary_with_carry_op_set = set(
    [
        HighLevelILOperation.HLIL_ADC,
        HighLevelILOperation.HLIL_SBB,
        HighLevelILOperation.HLIL_RLC,
        HighLevelILOperation.HLIL_RRC,
    ]
)


def parse_list(expr_list: list[binaryninja.HighLevelILInstruction]) -> list[Expr]:
    return [parse_ast(e) for e in expr_list]


def parse_ast(expr: binaryninja.HighLevelILInstruction) -> Expr:
    op = expr.operation
    address = expr.address
    instr_index = expr.instr_index
    expr_index = expr.expr_index
    args = [op, address, instr_index, expr_index]

    match op:
        case _ if op in unary_op_set:
            return parse_unary(expr, args)
        case _ if op in binary_op_set:
            return parse_binary(expr, args)
        case _ if op in binary_with_carry_op_set:
            return parse_binary_with_carry(expr, args)
        case HighLevelILOperation.HLIL_BLOCK:
            body = parse_list(expr.body)
            return Block(*args, body)
        case HighLevelILOperation.HLIL_IF:
            condition = parse_ast(expr.condition)
            true = parse_ast(expr.true)
            false = parse_ast(expr.false)
            return If(*args, condition, true, false)
        case HighLevelILOperation.HLIL_WHILE:
            condition = parse_ast(expr.condition)
            body = parse_ast(expr.body)
            return While(*args, condition, body)
        case HighLevelILOperation.HLIL_DO_WHILE:
            body = parse_ast(expr.body)
            condition = parse_ast(expr.condition)
            return DoWhile(*args, body, condition)
        case HighLevelILOperation.HLIL_FOR:
            init = parse_ast(expr.init)
            condition = parse_ast(expr.condition)
            update = parse_ast(expr.update)
            body = parse_ast(expr.body)
            return For(*args, init, condition, update, body)
        case HighLevelILOperation.HLIL_SWITCH:
            condition = parse_ast(expr.condition)
            default = parse_ast(expr.default)
            cases = parse_list(expr.cases)
            return Switch(*args, condition, default, cases)
        case HighLevelILOperation.HLIL_CASE:
            values = parse_list(expr.values)
            body = parse_ast(expr.body)
            return Case(*args, values, body)
        case HighLevelILOperation.HLIL_JUMP:
            dest = parse_ast(expr.dest)
            return Jump(*args, dest)
        case HighLevelILOperation.HLIL_RET:
            src = parse_list(expr.src)
            return Ret(*args, src)
        case HighLevelILOperation.HLIL_GOTO:
            target = parse_goto_label(expr.target)
            return Goto(*args, target)
        case HighLevelILOperation.HLIL_LABEL:
            target = parse_goto_label(expr.target)
            return Label(*args, target)
        case HighLevelILOperation.HLIL_VAR_DECLARE:
            var = expr.var.identifier
            return VarDeclare(*args, var)
        case HighLevelILOperation.HLIL_VAR_INIT:
            dest = expr.dest.identifier
            src = parse_ast(expr.src)
            return VarInit(*args, dest, src)
        case HighLevelILOperation.HLIL_ASSIGN:
            dest = parse_ast(expr.dest)
            src = parse_ast(expr.src)
            return Assign(*args, dest, src)
        case HighLevelILOperation.HLIL_ASSIGN_UNPACK:
            dest = parse_list(expr.dest)
            src = parse_ast(expr.src)
            return AssignUnpack(*args, dest, src)
        case HighLevelILOperation.HLIL_VAR:
            var = expr.var.identifier
            return Var(*args, var)
        case HighLevelILOperation.HLIL_STRUCT_FIELD:
            src = parse_ast(expr.src)
            offset = expr.offset
            member_index = expr.member_index
            return StructField(*args, src, offset, member_index)
        case HighLevelILOperation.HLIL_ARRAY_INDEX:
            src = parse_ast(expr.src)
            index = parse_ast(expr.index)
            return ArrayIndex(*args, src, index)
        case HighLevelILOperation.HLIL_SPLIT:
            high = parse_ast(expr.high)
            low = parse_ast(expr.low)
            return Split(*args, high, low)
        case HighLevelILOperation.HLIL_DEREF_FIELD:
            src = parse_ast(expr.src)
            offset = expr.offset
            member_index = expr.member_index
            return DerefField(*args, src, offset, member_index)
        case HighLevelILOperation.HLIL_CONST:
            constant = expr.constant
            return Const(*args, constant)
        case HighLevelILOperation.HLIL_CONST_DATA:
            constant_data = expr.constant_data.data.escape()
            return ConstData(*args, constant_data)
        case HighLevelILOperation.HLIL_CONST_PTR:
            constant = expr.constant
            return ConstPtr(*args, constant)
        case HighLevelILOperation.HLIL_EXTERN_PTR:
            constant = expr.constant
            offset = expr.offset
            return ExternPtr(*args, constant, offset)
        case HighLevelILOperation.HLIL_FLOAT_CONST:
            constant = expr.constant
            return FloatConst(*args, constant)
        case HighLevelILOperation.HLIL_IMPORT:
            constant = expr.constant
            return Import(*args, constant)
        case HighLevelILOperation.HLIL_CALL:
            dest = parse_ast(expr.dest)
            params = parse_list(expr.params)
            return Call(*args, dest, params)
        case HighLevelILOperation.HLIL_SYSCALL:
            params = parse_list(expr.params)
            return Syscall(*args, params)
        case HighLevelILOperation.HLIL_TAILCALL:
            dest = parse_ast(expr.dest)
            params = parse_list(expr.params)
            return Tailcall(*args, dest, params)
        case HighLevelILOperation.HLIL_INTRINSIC:
            intrinsic = parse_intrinsic(expr.intrinsic)
            params = parse_list(expr.params)
            return Intrinsic(*args, intrinsic, params)
        case HighLevelILOperation.HLIL_TRAP:
            vector = expr.vector
            return Trap(*args, vector)
        case (
            HighLevelILOperation.HLIL_NOP
            | HighLevelILOperation.HLIL_BREAK
            | HighLevelILOperation.HLIL_CONTINUE
            | HighLevelILOperation.HLIL_NORET
            | HighLevelILOperation.HLIL_BP
            | HighLevelILOperation.HLIL_UNDEF
            | HighLevelILOperation.HLIL_UNIMPL
            | HighLevelILOperation.HLIL_UNREACHABLE
        ):
            return Expr(*args)

    raise Exception(f"Patter matching does not exaustive. ({str(op)})")


def fold(f, init, expr):
    def helper(acc, elem):
        if isinstance(elem, Expr):
            return fold(f, acc, elem)
        return acc

    acc = f(init, expr)
    for field in fields(expr):
        value = getattr(expr, field.name)
        if isinstance(value, list):
            for elem in value:
                acc = helper(acc, elem)
        else:
            acc = helper(acc, value)

    return acc


def iter(f, expr):
    def helper(_, expr):
        f(expr)
        return None

    fold(helper, None, expr)

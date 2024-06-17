import binaryninja

from binaryninja import HighLevelILOperation
from dataclasses import dataclass, fields
from functools import partial

from .constants import *


@dataclass
class Expr:
    op: binaryninja.HighLevelILOperation
    address: int
    instr_index: binaryninja.InstructionIndex
    expr_index: binaryninja.ExpressionIndex


@dataclass
class Nop(Expr):
    pass


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
class Break(Expr):
    pass


@dataclass
class Continue(Expr):
    pass


@dataclass
class Jump(Expr):
    dest: Expr


@dataclass
class Ret(Expr):
    src: list[Expr]


@dataclass
class Noret(Expr):
    pass


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


def parse_unary(
    expr: binaryninja.HighLevelILInstruction, args, class_type: type[Unary]
) -> Unary:
    src = parse_ast(expr.src)
    return class_type(*args, src)


@dataclass
class Deref(Unary):
    pass


@dataclass
class DerefField(Expr):
    src: Expr
    offset: int
    member_index: int | None


@dataclass
class AddressOf(Unary):
    pass


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


def parse_binary(
    expr: binaryninja.HighLevelILInstruction, args, class_type: type[Binary]
) -> Binary:
    left = parse_ast(expr.left)
    right = parse_ast(expr.right)
    return class_type(*args, left, right)


@dataclass
class Add(Binary):
    pass


@dataclass
class BinaryWithCarry(Binary):
    carry: Expr


def parse_binary_with_carry(
    expr: binaryninja.HighLevelILInstruction, args, class_type: type[BinaryWithCarry]
) -> BinaryWithCarry:
    left = parse_ast(expr.left)
    right = parse_ast(expr.right)
    carry = parse_ast(expr.carry)
    return class_type(*args, left, right, carry)


@dataclass
class Adc(BinaryWithCarry):
    pass


@dataclass
class Sub(Binary):
    pass


@dataclass
class Sbb(BinaryWithCarry):
    pass


@dataclass
class And(Binary):
    pass


@dataclass
class Or(Binary):
    pass


@dataclass
class Xor(Binary):
    pass


@dataclass
class Lsl(Binary):
    pass


@dataclass
class Lsr(Binary):
    pass


@dataclass
class Asr(Binary):
    pass


@dataclass
class Rol(Binary):
    pass


@dataclass
class Rlc(BinaryWithCarry):
    pass


@dataclass
class Ror(Binary):
    pass


@dataclass
class Rrc(BinaryWithCarry):
    pass


@dataclass
class Mul(Binary):
    pass


@dataclass
class MuluDp(Binary):
    pass


@dataclass
class MulsDp(Binary):
    pass


@dataclass
class Divu(Binary):
    pass


@dataclass
class DivuDp(Binary):
    pass


@dataclass
class Divs(Binary):
    pass


@dataclass
class DivsDp(Binary):
    pass


@dataclass
class Modu(Binary):
    pass


@dataclass
class ModuDp(Binary):
    pass


@dataclass
class Mods(Binary):
    pass


@dataclass
class ModsDp(Binary):
    pass


@dataclass
class Neg(Unary):
    pass


@dataclass
class Not(Unary):
    pass


@dataclass
class Sx(Unary):
    pass


@dataclass
class Zx(Unary):
    pass


@dataclass
class LowPart(Unary):
    pass


@dataclass
class Call(Expr):
    dest: Expr
    params: list[Expr]


@dataclass
class CmpE(Binary):
    pass


@dataclass
class CmpNe(Binary):
    pass


@dataclass
class CmpSlt(Binary):
    pass


@dataclass
class CmpUlt(Binary):
    pass


@dataclass
class CmpSle(Binary):
    pass


@dataclass
class CmpUle(Binary):
    pass


@dataclass
class CmpSge(Binary):
    pass


@dataclass
class CmpUge(Binary):
    pass


@dataclass
class CmpSgt(Binary):
    pass


@dataclass
class CmpUgt(Binary):
    pass


@dataclass
class TestBit(Binary):
    pass


@dataclass
class BoolToInt(Unary):
    pass


@dataclass
class AddOverflow(Binary):
    pass


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
class Bp(Expr):
    pass


@dataclass
class Trap(Expr):
    vector: int


@dataclass
class Undef(Expr):
    pass


@dataclass
class Unimpl(Expr):
    pass


@dataclass
class UnimplMem(Unary):
    pass


@dataclass
class Fadd(Binary):
    pass


@dataclass
class Fsub(Binary):
    pass


@dataclass
class Fmul(Binary):
    pass


@dataclass
class Fdiv(Binary):
    pass


@dataclass
class Fsqrt(Unary):
    pass


@dataclass
class Fneg(Unary):
    pass


@dataclass
class Fabs(Unary):
    pass


@dataclass
class FloatToInt(Unary):
    pass


@dataclass
class IntToFloat(Unary):
    pass


@dataclass
class FloatConv(Unary):
    pass


@dataclass
class RoundToInt(Unary):
    pass


@dataclass
class Floor(Unary):
    pass


@dataclass
class Ceil(Unary):
    pass


@dataclass
class Ftrunc(Unary):
    pass


@dataclass
class FcmpE(Binary):
    pass


@dataclass
class FcmpNe(Binary):
    pass


@dataclass
class FcmpLt(Binary):
    pass


@dataclass
class FcmpLe(Binary):
    pass


@dataclass
class FcmpGe(Binary):
    pass


@dataclass
class FcmpGt(Binary):
    pass


@dataclass
class FcmpO(Binary):
    pass


@dataclass
class FcmpUo(Binary):
    pass


@dataclass
class Unreachable(Expr):
    pass


# Since we target HLIR, we didn't implemented class for *_SSA and *_PHI instructions.

unary_op_map = {
    HighLevelILOperation.HLIL_DEREF: Deref,
    HighLevelILOperation.HLIL_ADDRESS_OF: AddressOf,
    HighLevelILOperation.HLIL_NEG: Neg,
    HighLevelILOperation.HLIL_NOT: Not,
    HighLevelILOperation.HLIL_SX: Sx,
    HighLevelILOperation.HLIL_ZX: Zx,
    HighLevelILOperation.HLIL_LOW_PART: LowPart,
    HighLevelILOperation.HLIL_BOOL_TO_INT: BoolToInt,
    HighLevelILOperation.HLIL_UNIMPL_MEM: UnimplMem,
    HighLevelILOperation.HLIL_FSQRT: Fsqrt,
    HighLevelILOperation.HLIL_FNEG: Fneg,
    HighLevelILOperation.HLIL_FABS: Fabs,
    HighLevelILOperation.HLIL_FLOAT_TO_INT: FloatToInt,
    HighLevelILOperation.HLIL_INT_TO_FLOAT: IntToFloat,
    HighLevelILOperation.HLIL_FLOAT_CONV: FloatConv,
    HighLevelILOperation.HLIL_ROUND_TO_INT: RoundToInt,
    HighLevelILOperation.HLIL_FLOOR: Floor,
    HighLevelILOperation.HLIL_CEIL: Ceil,
    HighLevelILOperation.HLIL_FTRUNC: Ftrunc,
}

binary_op_map = {
    HighLevelILOperation.HLIL_ADD: Add,
    HighLevelILOperation.HLIL_SUB: Sub,
    HighLevelILOperation.HLIL_AND: And,
    HighLevelILOperation.HLIL_OR: Or,
    HighLevelILOperation.HLIL_XOR: Xor,
    HighLevelILOperation.HLIL_LSL: Lsl,
    HighLevelILOperation.HLIL_LSR: Lsr,
    HighLevelILOperation.HLIL_ASR: Asr,
    HighLevelILOperation.HLIL_ROL: Rol,
    HighLevelILOperation.HLIL_ROR: Ror,
    HighLevelILOperation.HLIL_MUL: Mul,
    HighLevelILOperation.HLIL_MULU_DP: MuluDp,
    HighLevelILOperation.HLIL_MULS_DP: MulsDp,
    HighLevelILOperation.HLIL_DIVU: Divu,
    HighLevelILOperation.HLIL_DIVU_DP: DivuDp,
    HighLevelILOperation.HLIL_DIVS: Divs,
    HighLevelILOperation.HLIL_DIVS_DP: DivsDp,
    HighLevelILOperation.HLIL_MODU: Modu,
    HighLevelILOperation.HLIL_MODU_DP: ModuDp,
    HighLevelILOperation.HLIL_MODS: Mods,
    HighLevelILOperation.HLIL_MODS_DP: ModsDp,
    HighLevelILOperation.HLIL_CMP_E: CmpE,
    HighLevelILOperation.HLIL_CMP_NE: CmpNe,
    HighLevelILOperation.HLIL_CMP_SLT: CmpSlt,
    HighLevelILOperation.HLIL_CMP_ULT: CmpUlt,
    HighLevelILOperation.HLIL_CMP_SLE: CmpSle,
    HighLevelILOperation.HLIL_CMP_ULE: CmpUle,
    HighLevelILOperation.HLIL_CMP_SGE: CmpSge,
    HighLevelILOperation.HLIL_CMP_UGE: CmpUge,
    HighLevelILOperation.HLIL_CMP_SGT: CmpSgt,
    HighLevelILOperation.HLIL_CMP_UGT: CmpUgt,
    HighLevelILOperation.HLIL_TEST_BIT: TestBit,
    HighLevelILOperation.HLIL_ADD_OVERFLOW: AddOverflow,
    HighLevelILOperation.HLIL_FADD: Fadd,
    HighLevelILOperation.HLIL_FSUB: Fsub,
    HighLevelILOperation.HLIL_FMUL: Fmul,
    HighLevelILOperation.HLIL_FDIV: Fdiv,
    HighLevelILOperation.HLIL_FCMP_E: FcmpE,
    HighLevelILOperation.HLIL_FCMP_NE: FcmpNe,
    HighLevelILOperation.HLIL_FCMP_LT: FcmpLt,
    HighLevelILOperation.HLIL_FCMP_LE: FcmpLe,
    HighLevelILOperation.HLIL_FCMP_GE: FcmpGe,
    HighLevelILOperation.HLIL_FCMP_GT: FcmpGt,
    HighLevelILOperation.HLIL_FCMP_O: FcmpO,
    HighLevelILOperation.HLIL_FCMP_UO: FcmpUo,
}

binary_with_carry_op_map = {
    HighLevelILOperation.HLIL_ADC: Adc,
    HighLevelILOperation.HLIL_SBB: Sbb,
    HighLevelILOperation.HLIL_RLC: Rlc,
    HighLevelILOperation.HLIL_RRC: Rrc,
}


def parse_list(expr_list: list[binaryninja.HighLevelILInstruction]) -> list[Expr]:
    return [parse_ast(e) for e in expr_list]


def parse_ast(expr: binaryninja.HighLevelILInstruction) -> Expr:
    op = expr.operation
    address = expr.address
    instr_index = expr.instr_index
    expr_index = expr.expr_index
    args = [op, address, instr_index, expr_index]

    parse_unary_helper = partial(parse_unary, expr, args)
    parse_binary_helper = partial(parse_binary, expr, args)
    parse_binary_with_carry_helper = partial(parse_binary_with_carry, expr, args)

    match op:
        case _ if op in unary_op_map:
            class_type = unary_op_map[op]
            return parse_unary_helper(class_type)
        case _ if op in binary_op_map:
            class_type = binary_op_map[op]
            return parse_binary_helper(class_type)
        case _ if op in binary_with_carry_op_map:
            class_type = binary_with_carry_op_map[op]
            return parse_binary_with_carry_helper(class_type)
        case HighLevelILOperation.HLIL_NOP:
            return Nop(*args)
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
        case HighLevelILOperation.HLIL_BREAK:
            return Break(*args)
        case HighLevelILOperation.HLIL_CONTINUE:
            return Continue(*args)
        case HighLevelILOperation.HLIL_JUMP:
            dest = parse_ast(expr.dest)
            return Jump(*args, dest)
        case HighLevelILOperation.HLIL_RET:
            src = parse_list(expr.src)
            return Ret(*args, src)
        case HighLevelILOperation.HLIL_NORET:
            return Noret(*args)
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
        case HighLevelILOperation.HLIL_BP:
            return Bp(*args)
        case HighLevelILOperation.HLIL_TRAP:
            vector = expr.vector
            return Trap(*args, vector)
        case HighLevelILOperation.HLIL_UNDEF:
            return Undef(*args)
        case HighLevelILOperation.HLIL_UNIMPL:
            return Unimpl(*args)
        case HighLevelILOperation.HLIL_UNREACHABLE:
            return Unreachable(*args)

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

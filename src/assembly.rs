#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

use crate::{
    ast::{self, Attributes::*, Namespace, StaticInit, Type, TypeCtx},
    ir,
};
use ecow::{EcoString as Ecow, eco_format};
use either::Either::{self, Left, Right};
use std::io::Write;

#[derive(Debug, Clone, Copy)]
enum BSymbol {
    Obj { type_: AsmType, is_static: bool },
    Func { _defined: bool },
}
impl From<TypeCtx> for BSymbol {
    fn from(value: TypeCtx) -> Self {
        let TypeCtx { type_, attr } = value;

        match (type_, attr) {
            (Type::Func { .. }, Func { defined, .. }) => BSymbol::Func { _defined: defined },
            (_, Func { .. }) | (Type::Func { .. }, _) => unreachable!(),

            (Type::Int | Type::UInt, Static { .. }) => {
                BSymbol::Obj { type_: Longword, is_static: true }
            }
            (Type::Int | Type::UInt, Local) => BSymbol::Obj { type_: Longword, is_static: false },

            (Type::Long | Type::ULong, Static { .. }) => {
                BSymbol::Obj { type_: Quadword, is_static: true }
            }
            (Type::Long | Type::ULong, Local) => BSymbol::Obj { type_: Quadword, is_static: false },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Program {
    top_level: Vec<TopLevel>,
    symbols: Namespace<BSymbol>,
}
impl Program {
    pub fn fixup_passes(&mut self) {
        self.replace_pseudos(&self.symbols.clone());
        self.adjust_instrs();
    }

    pub fn emit_code(&self, f: &mut impl Write) {
        self.top_level.iter().for_each(|tl| tl.emit_code(f));
    }

    fn replace_pseudos(&mut self, symbols: &Namespace<BSymbol>) {
        self.top_level.iter_mut().for_each(|tl| tl.replace_pseudos(symbols));
    }

    fn adjust_instrs(&mut self) {
        self.top_level.iter_mut().for_each(TopLevel::adjust_instrs);
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AsmType {
    Longword,
    Quadword,
}
use AsmType::*;

impl AsmType {
    fn width(self) -> u8 {
        match self {
            Longword => 4,
            Quadword => 8,
        }
    }

    fn emit_code(self) -> &'static str {
        match self {
            Longword => "l",
            Quadword => "q",
        }
    }
}

#[derive(Debug, Clone)]
pub enum TopLevel {
    Function { name: Ecow, instrs: Vec<Instr>, stack_size: i32, global: bool },
    StaticVariable { name: Ecow, global: bool, init: StaticInit, alignment: i32 },
}

impl TopLevel {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            TopLevel::Function { name, instrs, global, .. } => {
                if *global {
                    _ = writeln!(f, "\t.globl  _{name}");
                }
                _ = writeln!(f, "\t.text");
                _ = writeln!(f, "_{name}:");

                _ = writeln!(f, "\tpushq   %rbp");
                _ = writeln!(f, "\tmovq    %rsp,   %rbp");

                instrs.iter().for_each(|i| i.emit_code(f));
            }
            TopLevel::StaticVariable { name, global, init, alignment } if init.is_zero() => {
                if *global {
                    _ = writeln!(f, "\t.globl  _{name}");
                }
                _ = writeln!(f, "\t.bss");
                _ = writeln!(f, "\t.balign {alignment}");
                _ = writeln!(f, "_{name}:");
                _ = writeln!(f, "\t.zero   {alignment}");
            }
            TopLevel::StaticVariable { name, global, init, alignment } => {
                if *global {
                    _ = writeln!(f, "\t.globl  _{name}");
                }
                _ = writeln!(f, "\t.data");
                _ = writeln!(f, "\t.balign {alignment}");
                _ = writeln!(f, "_{name}:");
                _ = writeln!(f, "\t.{init}");
            }
        }
    }

    fn replace_pseudos(&mut self, symbols: &Namespace<BSymbol>) {
        match self {
            TopLevel::Function { instrs, stack_size, .. } => {
                let mut map = Namespace::default();
                let mut stack_depth = 0;

                instrs
                    .iter_mut()
                    .for_each(|inst| inst.replace_pseudos(&mut map, &mut stack_depth, symbols));

                *stack_size = stack_depth;
            }
            TopLevel::StaticVariable { .. } => {}
        }
    }

    #[allow(clippy::too_many_lines)]
    fn adjust_instrs(&mut self) {
        '_stack_frame: {
            match self {
                TopLevel::Function { instrs, stack_size, .. } => {
                    let sd = stack_size.unsigned_abs().next_multiple_of(16);
                    let instr = Instr::Binary(Operator::Sub, Quadword, Imm(sd.into()), Reg(SP, 8));

                    instrs.insert(0, instr);
                }
                TopLevel::StaticVariable { .. } => {}
            }
        }

        '_fixup_instrs: for _ in 0..3 {
            let TopLevel::Function { instrs, .. } = self else {
                break;
            };
            let mut out = Vec::with_capacity(instrs.len());

            for instr in std::mem::take(instrs) {
                match instr {
                    Instr::Mov(ty, Imm(0), dst @ Reg(..)) => {
                        out.push(Instr::Binary(Operator::Xor, ty, dst.clone(), dst));
                    }
                    Instr::Mov(ty, src @ (Stack(_) | Data(_)), dst @ (Stack(_) | Data(_))) => {
                        out.extend([
                            Instr::Mov(ty, src, Reg(R10, ty.width())),
                            Instr::Mov(ty, Reg(R10, ty.width()), dst),
                        ]);
                    }
                    Instr::Mov(ty, src @ Imm(i), dst @ (Stack(_) | Data(_)))
                        if i32::try_from(i).is_err() =>
                    {
                        out.extend([
                            Instr::Mov(ty, src, Reg(R10, ty.width())),
                            Instr::Mov(ty, Reg(R10, ty.width()), dst),
                        ]);
                    }
                    Instr::Mov(Longword, Imm(i), dst) if i32::try_from(i).is_err() => {
                        out.push(Instr::Mov(Longword, Imm(i64::from(i as i32)), dst));
                    }
                    Instr::Cmp(ty, src @ Imm(i), dst)
                        if i32::try_from(i).is_err() =>
                    {
                        out.extend([
                            Instr::Mov(ty, src, Reg(R10, ty.width())),
                            Instr::Cmp(ty, Reg(R10, ty.width()), dst),
                        ]);
                    }
                    Instr::Cmp(ty, src @ (Stack(_) | Data(_)), dst @ (Stack(_) | Data(_))) => {
                        out.extend([
                            Instr::Mov(ty, src, Reg(R10, ty.width())),
                            Instr::Cmp(ty, Reg(R10, ty.width()), dst),
                        ]);
                    }
                    Instr::Cmp(ty, src, dst @ Imm(_)) => {
                        out.extend([
                            Instr::Mov(ty, dst, Reg(R11, ty.width())),
                            Instr::Cmp(ty, src, Reg(R11, ty.width())),
                        ]);
                    }
                    Instr::Push(src @ Imm(i)) if i32::try_from(i).is_err() => {
                        out.extend([
                            Instr::Mov(Quadword, src, Reg(R10, 8)),
                            Instr::Push(Reg(R10, 8)),
                        ]);
                    }
                    Instr::Idiv(ty, v @ Imm(_)) => {
                        out.extend([
                            Instr::Mov(ty, v, Reg(R10, ty.width())),
                            Instr::Idiv(ty, Reg(R10, ty.width())),
                        ]);
                    }
                    Instr::Binary(
                        opp @ (Operator::Add
                        | Operator::Sub
                        | Operator::Mul
                        | Operator::And
                        | Operator::Or
                        | Operator::Xor
                        // maybe ?
                        | Operator::Shr
                        | Operator::Shl),
                        ty,
                        src @ Imm(i),
                        dst,
                    ) if i32::try_from(i).is_err() => out.extend([
                        Instr::Mov(ty, src, Reg(R10, ty.width())),
                        Instr::Binary(opp, ty, Reg(R10, ty.width()), dst),
                    ]),
                    Instr::Binary(
                        opp @ (Operator::Add
                        | Operator::Sub
                        | Operator::And
                        | Operator::Or
                        | Operator::Xor),
                        ty,
                        src @ (Stack(_) | Data(_)),
                        dst @ (Stack(_) | Data(_)),
                    ) => out.extend([
                        Instr::Mov(ty, src, Reg(R10, ty.width())),
                        Instr::Binary(opp, ty, Reg(R10, ty.width()), dst),
                    ]),

                    Instr::Binary(Operator::Mul, ty, src, dst @ (Stack(_) | Data(_))) => {
                        out.extend([
                            Instr::Mov(ty, dst.clone(), Reg(R11, ty.width())),
                            Instr::Binary(Operator::Mul, ty, src, Reg(R11, ty.width())),
                            Instr::Mov(ty, Reg(R11, ty.width()), dst),
                        ]);
                    }
                    Instr::Binary(
                        opp @ (Operator::Shl | Operator::Shr),
                        ty,
                        src,
                        dst @ (Stack(_) | Data(_)),
                    ) if src != Reg(CX, 1) => {
                        out.extend([
                            Instr::Mov(ty, src, Reg(CX, ty.width())),
                            Instr::Binary(opp, ty, Reg(CX, 1), dst),
                        ]);
                    }
                    Instr::Movsx(src @ Imm(_), dst) => {
                        // maybe it is better to split it in two passes?
                        out.extend([
                            Instr::Mov(Longword, src, Reg(R10, 4)),
                            Instr::Movsx(Reg(R10, 4), dst),
                        ]);
                    }
                    Instr::Movsx(src, dst @ (Stack(_) | Data(_))) => {
                        out.extend([
                            Instr::Movsx(src, Reg(R11, 8)),
                            Instr::Mov(Quadword, Reg(R11, 8), dst),
                        ]);
                    }
                    other => out.push(other),
                }
            }

            *instrs = out;
        }
    }
}

#[derive(Debug, Clone)]
pub enum Instr {
    Mov(AsmType, Operand, Operand),
    Movsx(Operand, Operand),
    Unary(Operator, AsmType, Operand),
    Binary(Operator, AsmType, Operand, Operand),
    Cmp(AsmType, Operand, Operand),
    Idiv(AsmType, Operand),
    Cdq(AsmType),
    Jmp(Ecow),
    JmpCC(CondCode, Ecow),
    SetCC(CondCode, Operand),
    Label(Ecow),
    Push(Operand),
    Call(Ecow),
    Ret,
}
impl Instr {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Self::Mov(ty, src, dst) => {
                let ty = ty.emit_code();
                let src = src.emit_code() + ",";
                let dst = dst.emit_code();

                _ = writeln!(f, "\tmov{ty}    {src:<7} {dst}");
            }
            Self::Unary(un_op, ty, operand) => {
                let ty = ty.emit_code();
                let uo = un_op.emit_code();
                let uo = eco_format!("{uo}{ty}");
                let op = operand.emit_code();

                _ = writeln!(f, "\t{uo:<7} {op}");
            }
            Self::Binary(bin_op, ty, op1, op2) => {
                let ty = ty.emit_code();
                let bo = bin_op.emit_code();
                let bo = eco_format!("{bo}{ty}");

                let o1 = op1.emit_code() + ",";
                let o2 = op2.emit_code();

                _ = writeln!(f, "\t{bo:<7} {o1:<7} {o2}");
            }
            Self::Idiv(ty, op) => {
                let ty = ty.emit_code();
                _ = writeln!(f, "\tidiv{ty}    {}", op.emit_code());
            }

            Self::Cdq(ty) => match ty {
                Longword => _ = writeln!(f, "\tcdq"),
                Quadword => _ = writeln!(f, "\tcqo"),
            },

            Self::Push(op) => _ = writeln!(f, "\tpushq   {}", op.emit_code()),
            Self::Call(label) => _ = writeln!(f, "\tcall    _{label}"),

            Self::Ret => {
                _ = writeln!(f, "\tmovq    %rbp,   %rsp");
                _ = writeln!(f, "\tpopq    %rbp");
                _ = writeln!(f, "\tret");
            }
            Self::Cmp(ty, op1, op2) => {
                let ty = ty.emit_code();
                let op1 = op1.emit_code() + ",";
                let op2 = op2.emit_code();

                _ = writeln!(f, "\tcmp{ty}    {op1:<7} {op2}");
            }
            Self::SetCC(cond, op) => {
                let cond = cond.emit_code();
                let op = op.emit_code();

                _ = writeln!(f, "\tset{cond:<4} {op}");
            }
            Self::Jmp(label) => _ = writeln!(f, "\tjmp     .L{label}"),
            Self::JmpCC(cond, label) => {
                let cond = cond.emit_code();
                _ = writeln!(f, "\tj{cond:<6} .L{label}");
            }
            Self::Label(label) => _ = writeln!(f, "    .L{label}:"),
            Self::Movsx(src, dst) => {
                let src = src.emit_code() + ",";
                let dst = dst.emit_code();

                _ = writeln!(f, "\tmovslq  {src:<7} {dst}");
            }
        }
    }

    fn replace_pseudos(
        &mut self,
        map: &mut Namespace<i32>,
        stack_depth: &mut i32,
        symbols: &Namespace<BSymbol>,
    ) {
        match self {
            Self::Unary(_, _, opp) | Self::Idiv(_, opp) | Self::SetCC(_, opp) => {
                opp.replace_pseudos(map, stack_depth, symbols);
            }
            Self::Binary(_, _, opp1, opp2)
            | Self::Mov(_, opp1, opp2)
            | Self::Cmp(_, opp1, opp2) => {
                opp1.replace_pseudos(map, stack_depth, symbols);
                opp2.replace_pseudos(map, stack_depth, symbols);
            }
            Self::Push(opp) => opp.replace_pseudos(map, stack_depth, symbols),

            Self::Movsx(op1, op2) => {
                op1.replace_pseudos(map, stack_depth, symbols);
                op2.replace_pseudos(map, stack_depth, symbols);
            }

            Self::Cdq(_)
            | Self::Ret
            | Self::Jmp(..)
            | Self::JmpCC(..)
            | Self::Label(..)
            | Self::Call(_) => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    Imm(i64),
    Reg(Register, u8),
    Pseudo(Ecow),
    Stack(i32),
    Data(Ecow),
}
use Operand::*;
impl Operand {
    fn align_width(self, ty: AsmType) -> Self {
        match (self, ty) {
            (Reg(r, _), Longword) => Reg(r, 4),
            (Reg(r, _), Quadword) => Reg(r, 8),
            (any, _) => any,
        }
    }

    fn emit_code(&self) -> Ecow {
        match self {
            Imm(i) => eco_format!("${i}"),
            Reg(r, s) => r.emit_code(*s).into(),
            Stack(i) => eco_format!("{i}(%rbp)"),
            Data(name) => eco_format!("_{name}(%rip)"),

            Pseudo(e) => unreachable!("pseudo register {e} printed"),
        }
    }

    fn replace_pseudos(
        &mut self,
        map: &mut Namespace<i32>,
        stack_depth: &mut i32,
        symbols: &Namespace<BSymbol>,
    ) {
        let Pseudo(name) = self else {
            return;
        };

        let mut new_place = if map.contains_key(name) {
            Stack(map[name])
        } else if symbols
            .get(name)
            .is_some_and(|tc| matches!(tc, BSymbol::Obj { is_static: true, .. }))
        {
            Data(name.clone())
        } else {
            let Some(BSymbol::Obj { type_, .. }) = symbols.get(&*name) else { unreachable!() };

            let curr_depth = *stack_depth;
            match (curr_depth.abs() % 8, type_) {
                (0, Quadword) => *stack_depth -= 8,
                (v, Quadword) => *stack_depth -= 8 + (8 - v),
                (_, Longword) => *stack_depth -= 4,
            }
            // do i need this assertion?
            match type_ {
                Longword => assert!(stack_depth.abs() % 4 == 0),
                Quadword => assert!(stack_depth.abs() % 8 == 0),
            }

            map.insert(name.clone(), *stack_depth);
            Stack(*stack_depth)
        };

        std::mem::swap(self, &mut new_place);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    // unary
    Not,
    Neg,
    // binary
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shr,
    Shl,
}
impl Operator {
    fn emit_code(self) -> &'static str {
        match self {
            Self::Not => "not",
            Self::Neg => "neg",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "imul",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Shr => "sar",
            Self::Shl => "shl",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Register {
    AX,
    DX,
    DI,
    SI,
    R8,
    R9,
    R10,
    R11,
    CX,
    SP,
}
const ARG_REGISTERS: [Register; 6] = [DI, SI, DX, CX, R8, R9];
use Register::*;
impl Register {
    #[rustfmt::skip]
    fn emit_code(self, byte_size: u8) -> &'static str {
        assert!(byte_size == 1 || byte_size == 2 || byte_size == 4 || byte_size == 8);
        match (self, byte_size) {
            (AX, 1) => "%al",   (AX, 2) => "%ax",   (AX, 4) => "%eax",  (AX, 8) => "%rax",
            (DX, 1) => "%dl",   (DX, 2) => "%dx",   (DX, 4) => "%edx",  (DX, 8) => "%rdx",
            (DI, 1) => "%dil",  (DI, 2) => "%di",   (DI, 4) => "%edi",  (DI, 8) => "%rdi",
            (SI, 1) => "%sil",  (SI, 2) => "%si",   (SI, 4) => "%esi",  (SI, 8) => "%rsi",
            (R8, 1) => "%r8b",  (R8, 2) => "%r8w",  (R8, 4) => "%r8d",  (R8, 8) => "%r8",
            (R9, 1) => "%r9b",  (R9, 2) => "%r9w",  (R9, 4) => "%r9d",  (R9, 8) => "%r9",
            (R10,1) => "%r10b", (R10,2) => "%r10w", (R10,4) => "%r10d", (R10,8) => "%r10",
            (R11,1) => "%r11b", (R11,2) => "%r11w", (R11,4) => "%r11d", (R11,8) => "%r11",
            (CX, 1) => "%cl",   (CX, 2) => "%cx",   (CX, 4) => "%ecx",  (CX, 8) => "%rcx",
       /* stack pointer */      (SP, 2) => "%sp",   (SP, 4) => "%esp",  (SP, 8) => "%rsp",
            _ => unreachable!()
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CondCode {
    E,
    NE,
    L,
    LE,
    G,
    GE,
}
use CondCode::*;
impl CondCode {
    fn emit_code(self) -> &'static str {
        match self {
            Self::E => "e",
            Self::NE => "ne",
            Self::L => "l",
            Self::LE => "le",
            Self::G => "g",
            Self::GE => "ge",
        }
    }
}

// =========

impl ir::Program {
    pub fn to_asm(&self) -> Program {
        // different order from the book
        let symbols = self.symbols.iter().map(|(k, v)| (k.clone(), v.clone().into())).collect();
        let top_level = self.top_level.iter().map(|tl| tl.to_asm(&symbols)).collect();

        Program { top_level, symbols }
    }
}

impl ir::TopLevel {
    fn to_asm(&self, symbols: &Namespace<BSymbol>) -> TopLevel {
        match self {
            ir::TopLevel::Function { name, global, params, body } => {
                let instrs = ARG_REGISTERS
                    .into_iter()
                    .map(|r| Reg(r, 0))
                    .chain((16..).step_by(8).map(Stack))
                    .zip(params.clone())
                    .map(|(src, param)| {
                        let Some(BSymbol::Obj { type_, .. }) = symbols.get(&param) else {
                            unreachable!()
                        };

                        Instr::Mov(*type_, src.align_width(*type_), Pseudo(param))
                    })
                    .chain(body.iter().flat_map(|e| e.to_asm(symbols)))
                    .collect();

                TopLevel::Function { name: name.clone(), instrs, stack_size: 0, global: *global }
            }
            ir::TopLevel::StaticVar { name, global, type_, init } => TopLevel::StaticVariable {
                name: name.clone(),
                global: *global,
                init: *init,
                alignment: match type_ {
                    Type::Int | Type::UInt => 4,
                    Type::Long | Type::ULong => 8,
                    Type::Func { .. } => unreachable!(),
                },
            },
        }
    }
}

impl ir::Instr {
    #[allow(clippy::too_many_lines)]
    fn to_asm(&self, symbols: &Namespace<BSymbol>) -> Vec<Instr> {
        match self {
            Self::Return(value) => {
                let ty = value.to_asm_type(symbols);
                vec![
                    Instr::Mov(ty, value.to_asm(), Operand::Reg(Register::AX, ty.width())),
                    Instr::Ret,
                ]
            }

            Self::SignExtend { src, dst } => vec![Instr::Movsx(src.to_asm(), dst.to_asm())],
            Self::Truncate { src, dst } => vec![Instr::Mov(Longword, src.to_asm(), dst.to_asm())],
            Self::ZeroExtend { .. } => todo!(),

            Self::Unary { op: ir::UnOp::Not, src, dst } => {
                let src_ty = src.to_asm_type(symbols);
                let Some(BSymbol::Obj { type_: dst_ty, .. }) = symbols.get(&dst.0) else {
                    unreachable!()
                };
                let dst = dst.to_asm();

                vec![
                    Instr::Cmp(src_ty, Operand::Imm(0), src.to_asm()),
                    Instr::Mov(*dst_ty, Operand::Imm(0), dst.clone()),
                    Instr::SetCC(E, dst),
                ]
            }
            Self::Unary { op, src, dst } => {
                let src_ty = src.to_asm_type(symbols);
                let dst = dst.to_asm();
                vec![
                    Instr::Mov(src_ty, src.to_asm(), dst.clone()),
                    Instr::Unary(op.to_asm(), src_ty, dst),
                ]
            }
            Self::Binary { op, lhs, rhs, dst } => {
                let src_ty = lhs.to_asm_type(symbols);
                match op {
                    ir::BinOp::Add
                    | ir::BinOp::Subtract
                    | ir::BinOp::Multiply
                    | ir::BinOp::BitAnd
                    | ir::BinOp::BitOr
                    | ir::BinOp::BitXor
                    | ir::BinOp::LeftShift
                    | ir::BinOp::RightShift => {
                        let dst = dst.to_asm();
                        vec![
                            Instr::Mov(src_ty, lhs.to_asm(), dst.clone()),
                            Instr::Binary(op.to_asm().unwrap_left(), src_ty, rhs.to_asm(), dst),
                        ]
                    }
                    ir::BinOp::Divide | ir::BinOp::Reminder => {
                        let res = match op {
                            ir::BinOp::Divide => Register::AX,
                            ir::BinOp::Reminder => Register::DX,
                            _ => unreachable!(),
                        };
                        vec![
                            Instr::Mov(
                                src_ty,
                                lhs.to_asm(),
                                Operand::Reg(Register::AX, 4).align_width(src_ty),
                            ),
                            Instr::Cdq(src_ty),
                            Instr::Idiv(src_ty, rhs.to_asm()),
                            Instr::Mov(
                                src_ty,
                                Operand::Reg(res, 4).align_width(src_ty),
                                dst.to_asm(),
                            ),
                        ]
                    }
                    ir::BinOp::Equal
                    | ir::BinOp::NotEqual
                    | ir::BinOp::LessThan
                    | ir::BinOp::LessOrEqual
                    | ir::BinOp::GreaterThan
                    | ir::BinOp::GreaterOrEqual => {
                        let Some(BSymbol::Obj { type_: dst_ty, .. }) = symbols.get(&dst.0) else {
                            unreachable!()
                        };
                        let dst = dst.to_asm();

                        vec![
                            Instr::Cmp(src_ty, rhs.to_asm(), lhs.to_asm()),
                            Instr::Mov(*dst_ty, Operand::Imm(0), dst.clone()),
                            Instr::SetCC(op.to_asm().unwrap_right(), dst),
                        ]
                    }
                }
            }
            Self::Copy { src, dst } => {
                let src_ty = src.to_asm_type(symbols);
                vec![Instr::Mov(src_ty, src.to_asm(), dst.to_asm())]
            }
            Self::Jump { target } => vec![Instr::Jmp(target.clone())],
            Self::JumpIfZero { cond, target } => {
                let cond_ty = cond.to_asm_type(symbols);
                vec![
                    Instr::Cmp(cond_ty, Operand::Imm(0), cond.to_asm()),
                    Instr::JmpCC(E, target.clone()),
                ]
            }
            Self::JumpIfNotZero { cond, target } => {
                let cond_ty = cond.to_asm_type(symbols);
                vec![
                    Instr::Cmp(cond_ty, Operand::Imm(0), cond.to_asm()),
                    Instr::JmpCC(NE, target.clone()),
                ]
            }
            Self::Label(name) => vec![Instr::Label(name.clone())],
            Self::FuncCall { name, args, dst } => {
                let Some(BSymbol::Obj { type_: dst_ty, .. }) = symbols.get(&dst.0) else {
                    unreachable!()
                };
                let (reg_args, stack_args) =
                    args.split_at_checked(6).unwrap_or((args.as_slice(), &[]));

                (stack_args.len() % 2 != 0)
                    .then_some(Instr::Binary(Operator::Sub, Quadword, Imm(8), Reg(SP, 8)))
                    .into_iter()
                    .chain(ARG_REGISTERS.into_iter().zip(reg_args).map(|(reg, arg)| {
                        let arg_ty = arg.to_asm_type(symbols);
                        Instr::Mov(arg_ty, arg.to_asm(), Reg(reg, 4).align_width(arg_ty))
                    }))
                    .chain(stack_args.iter().rev().flat_map(|arg| {
                        let arg_ty = arg.to_asm_type(symbols);

                        match arg.to_asm() {
                            opp @ Imm(_) => vec![Instr::Push(opp)],
                            Reg(r, _) => vec![Instr::Push(Reg(r, 8))],
                            opp => vec![
                                Instr::Mov(arg_ty, opp, Reg(AX, 0).align_width(arg_ty)),
                                Instr::Push(Reg(AX, 8)),
                            ],
                        }
                    }))
                    .chain([Instr::Call(name.clone())])
                    .chain((!stack_args.is_empty()).then_some(Instr::Binary(
                        Operator::Add,
                        Quadword,
                        Imm(8 * stack_args.len() as i64
                            + (stack_args.len() % 2 != 0).then_some(8).unwrap_or_default()),
                        Reg(SP, 8),
                    )))
                    .chain([Instr::Mov(*dst_ty, Reg(AX, dst_ty.width()), dst.to_asm())])
                    .collect()
            }
        }
    }
}
impl ir::Value {
    fn to_asm(&self) -> Operand {
        match self {
            Self::Const(ast::Const::Int(i)) => Operand::Imm(i64::from(*i)),
            Self::Const(ast::Const::Long(i)) => Operand::Imm(*i),
            Self::Var(place) => place.to_asm(),

            Self::Const(ast::Const::UInt(_) | ast::Const::ULong(_)) => todo!(),
        }
    }
    fn to_asm_type(&self, symbols: &Namespace<BSymbol>) -> AsmType {
        match self {
            ir::Value::Const(c) => c.to_asm_type(),

            ir::Value::Var(ir::Place(ident)) => match symbols.get(ident) {
                Some(BSymbol::Obj { type_, .. }) => *type_,
                e => unreachable!("{e:?}"),
            },
        }
    }
}
impl ast::Const {
    fn to_asm_type(self) -> AsmType {
        match self {
            ast::Const::Int(_) | ast::Const::UInt(_) => Longword,
            ast::Const::Long(_) | ast::Const::ULong(_) => Quadword,
        }
    }
}
impl ir::Place {
    fn to_asm(&self) -> Operand {
        Operand::Pseudo(self.0.clone())
    }
}
impl ir::UnOp {
    fn to_asm(self) -> Operator {
        match self {
            Self::Complement => Operator::Not,
            Self::Negate => Operator::Neg,
            Self::Not => unreachable!("Not is implemented otherwise"),
        }
    }
}
impl ir::BinOp {
    fn to_asm(self) -> Either<Operator, CondCode> {
        match self {
            Self::Add => Left(Operator::Add),
            Self::Subtract => Left(Operator::Sub),
            Self::Multiply => Left(Operator::Mul),
            Self::BitAnd => Left(Operator::And),
            Self::BitOr => Left(Operator::Or),
            Self::BitXor => Left(Operator::Xor),
            Self::LeftShift => Left(Operator::Shl),
            Self::RightShift => Left(Operator::Shr),

            Self::Divide | Self::Reminder => {
                unreachable!("Divide and Reminder are implemented in other ways")
            }

            Self::Equal => Right(CondCode::E),
            Self::NotEqual => Right(CondCode::NE),
            Self::LessThan => Right(CondCode::L),
            Self::LessOrEqual => Right(CondCode::LE),
            Self::GreaterThan => Right(CondCode::G),
            Self::GreaterOrEqual => Right(CondCode::GE),
        }
    }
}

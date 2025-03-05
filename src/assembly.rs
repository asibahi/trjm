#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

use crate::{
    ast::{Attributes, Namespace, TypeCtx},
    ir,
};
use ecow::{EcoString as Ecow, eco_format};
use either::Either::{self, Left, Right};
use std::io::Write;

#[derive(Debug, Clone)]
pub struct Program {
    top_level: Vec<TopLevel>,
    symbols: Namespace<TypeCtx>,
}
impl Program {
    pub fn fixup_passes(&mut self) {
        self.replace_pseudos(&self.symbols.clone());
        self.adjust_instrs();
    }

    pub fn emit_code(&self, f: &mut impl Write) {
        self.top_level.iter().for_each(|tl| tl.emit_code(f));
    }

    fn replace_pseudos(&mut self, symbols: &Namespace<TypeCtx>) {
        self.top_level.iter_mut().for_each(|tl| tl.replace_pseudos(symbols));
    }

    fn adjust_instrs(&mut self) {
        self.top_level.iter_mut().for_each(TopLevel::adjust_instrs);
    }
}

#[derive(Debug, Clone)]
pub enum TopLevel {
    Function { name: Ecow, instrs: Vec<Instr>, stack_size: i32, global: bool },
    StaticVariable { name: Ecow, global: bool, init: i32 },
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
            TopLevel::StaticVariable { name, global, init } => {
                if *global {
                    _ = writeln!(f, "\t.globl  _{name}");
                }
                if *init == 0 {
                    _ = writeln!(f, "\t.bss");
                } else {
                    _ = writeln!(f, "\t.data");
                }
                _ = writeln!(f, "\t.balign 4");
                _ = writeln!(f, "_{name}:");
                if *init == 0 {
                    _ = writeln!(f, "\t.zero   4");
                } else {
                    _ = writeln!(f, "\t.long   {init}");
                }
            }
        }
    }

    fn replace_pseudos(&mut self, symbols: &Namespace<TypeCtx>) {
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

    fn adjust_instrs(&mut self) {
        '_stack_frame: {
            match self {
                TopLevel::Function { instrs, stack_size, .. } => {
                    let sd = stack_size.unsigned_abs().next_multiple_of(16);

                    instrs.insert(0, Instr::AllocateStack(sd));
                }
                TopLevel::StaticVariable { .. } => {}
            }
        }

        '_fixup_instrs: for _ in 0..2 {
            let TopLevel::Function { instrs, .. } = self else {
                break;
            };
            let mut out = Vec::with_capacity(instrs.len());

            for instr in std::mem::take(instrs) {
                match instr {
                    Instr::Mov(Imm(0), dst @ Reg(..)) => {
                        out.push(Instr::Binary(Operator::Xor, dst.clone(), dst));
                    }
                    Instr::Mov(src @ (Stack(_) | Data(_)), dst @ (Stack(_) | Data(_))) => {
                        out.extend([Instr::Mov(src, Reg(R10, 4)), Instr::Mov(Reg(R10, 4), dst)]);
                    }
                    Instr::Cmp(src @ (Stack(_) | Data(_)), dst @ (Stack(_) | Data(_))) => {
                        out.extend([Instr::Mov(src, Reg(R10, 4)), Instr::Cmp(Reg(R10, 4), dst)]);
                    }
                    Instr::Cmp(src, dst @ Imm(_)) => {
                        out.extend([Instr::Mov(dst, Reg(R11, 4)), Instr::Cmp(src, Reg(R11, 4))]);
                    }
                    Instr::Idiv(v @ Imm(_)) => {
                        out.extend([Instr::Mov(v, Reg(R10, 4)), Instr::Idiv(Reg(R10, 4))]);
                    }
                    Instr::Binary(
                        opp @ (Operator::Add
                        | Operator::Sub
                        | Operator::And
                        | Operator::Or
                        | Operator::Xor),
                        src @ (Stack(_) | Data(_)),
                        dst @ (Stack(_) | Data(_)),
                    ) => out.extend([
                        Instr::Mov(src, Reg(R10, 4)),
                        Instr::Binary(opp, Reg(R10, 4), dst),
                    ]),
                    Instr::Binary(Operator::Mul, src, dst @ (Stack(_) | Data(_))) => out.extend([
                        Instr::Mov(dst.clone(), Reg(R11, 4)),
                        Instr::Binary(Operator::Mul, src, Reg(R11, 4)),
                        Instr::Mov(Reg(R11, 4), dst),
                    ]),
                    Instr::Binary(
                        opp @ (Operator::Shl | Operator::Shr),
                        src,
                        dst @ (Stack(_) | Data(_)),
                    ) if src != Reg(CX, 1) => {
                        out.extend([
                            Instr::Mov(src, Reg(CX, 4)),
                            Instr::Binary(opp, Reg(CX, 1), dst),
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
    Mov(Operand, Operand),
    Unary(Operator, Operand),
    Binary(Operator, Operand, Operand),
    Cmp(Operand, Operand),
    Idiv(Operand),
    Cdq,
    Jmp(Ecow),
    JmpCC(CondCode, Ecow),
    SetCC(CondCode, Operand),
    Label(Ecow),
    AllocateStack(u32),
    DeallocateStack(u32),
    Push(Operand),
    Call(Ecow),
    Ret,
}
impl Instr {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Self::Mov(src, dst) => {
                let src = src.emit_code() + ",";
                let dst = dst.emit_code();

                _ = writeln!(f, "\tmovl    {src:<7} {dst}");
            }
            Self::Unary(un_op, operand) => {
                let uo = un_op.emit_code();
                let op = operand.emit_code();

                _ = writeln!(f, "\t{uo:<7} {op}");
            }
            Self::Binary(bin_op, op1, op2) => {
                let bo = bin_op.emit_code();
                let o1 = op1.emit_code() + ",";
                let o2 = op2.emit_code();

                _ = writeln!(f, "\t{bo:<7} {o1:<7} {o2}");
            }
            Self::Idiv(op) => _ = writeln!(f, "\tidivl    {}", op.emit_code()),

            Self::Cdq => _ = writeln!(f, "\tcdq"),

            Self::AllocateStack(i) => {
                _ = writeln!(f, "\tsubq    {:<7} %rsp", Imm(*i as i32).emit_code() + ",");
            }
            Self::DeallocateStack(i) => {
                _ = writeln!(f, "\taddq    {:<7} %rsp", Imm(*i as i32).emit_code() + ",");
            }

            Self::Push(op) => _ = writeln!(f, "\tpushq   {}", op.emit_code()),
            Self::Call(label) => _ = writeln!(f, "\tcall    _{label}"),

            Self::Ret => {
                _ = writeln!(f, "\tmovq    %rbp,   %rsp");
                _ = writeln!(f, "\tpopq    %rbp");
                _ = writeln!(f, "\tret");
            }
            Self::Cmp(op1, op2) => {
                let op1 = op1.emit_code() + ",";
                let op2 = op2.emit_code();

                _ = writeln!(f, "\tcmpl    {op1:<7} {op2}");
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
            Self::Label(label) => _ = writeln!(f, ".L{label}:"),
        }
    }

    fn replace_pseudos(
        &mut self,
        map: &mut Namespace<i32>,
        stack_depth: &mut i32,
        symbols: &Namespace<TypeCtx>,
    ) {
        match self {
            Self::Unary(_, opp) | Self::Idiv(opp) | Self::SetCC(_, opp) => {
                opp.replace_pseudos(map, stack_depth, symbols);
            }
            Self::Binary(_, opp1, opp2) | Self::Mov(opp1, opp2) | Self::Cmp(opp1, opp2) => {
                opp1.replace_pseudos(map, stack_depth, symbols);
                opp2.replace_pseudos(map, stack_depth, symbols);
            }
            Self::Push(opp) => opp.replace_pseudos(map, stack_depth, symbols),

            Self::Cdq
            | Self::AllocateStack(_)
            | Self::DeallocateStack(_)
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
    Imm(i32),
    Reg(Register, usize),
    Pseudo(Ecow),
    Stack(i32),
    Data(Ecow),
}
use Operand::*;
impl Operand {
    fn emit_code(&self) -> Ecow {
        match self {
            Imm(i) => eco_format!("${i}"),
            Reg(r, s) => r.emit_code(*s),
            Stack(i) => eco_format!("{i}(%rbp)"),
            Data(name) => eco_format!("_{name}(%rip)"),

            Pseudo(e) => unreachable!("pseudo register {e} printed"),
        }
    }

    fn replace_pseudos(
        &mut self,
        map: &mut Namespace<i32>,
        stack_depth: &mut i32,
        symbols: &Namespace<TypeCtx>,
    ) {
        let Pseudo(name) = self else {
            return;
        };

        let mut new_place = if map.contains_key(name) {
            Stack(map[name])
        } else if symbols.get(name).is_some_and(|tc| matches!(tc.attr, Attributes::Static { .. })) {
            Data(name.clone())
        } else {
            *stack_depth -= 4;
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
    fn emit_code(self) -> Ecow {
        match self {
            Self::Not => eco_format!("notl"),
            Self::Neg => eco_format!("negl"),
            Self::Add => eco_format!("addl"),
            Self::Sub => eco_format!("subl"),
            Self::Mul => eco_format!("imull"),
            Self::And => eco_format!("andl"),
            Self::Or => eco_format!("orl"),
            Self::Xor => eco_format!("xorl"),
            Self::Shr => eco_format!("sarl"),
            Self::Shl => eco_format!("shll"),
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
}
const ARG_REGISTERS: [Register; 6] = [DI, SI, DX, CX, R8, R9];
use Register::*;
impl Register {
    fn emit_code(self, byte_size: usize) -> Ecow {
        assert!(byte_size == 1 || byte_size == 2 || byte_size == 4 || byte_size == 8);
        match (self, byte_size) {
            (AX, 1) => Ecow::from("%al"),
            (AX, 2) => Ecow::from("%ax"),
            (AX, 4) => Ecow::from("%eax"),
            (AX, _) => Ecow::from("%rax"),

            (DX, 1) => Ecow::from("%dl"),
            (DX, 2) => Ecow::from("%dx"),
            (DX, 4) => Ecow::from("%edx"),
            (DX, _) => Ecow::from("%rdx"),

            (DI, 1) => Ecow::from("%dil"),
            (DI, 2) => Ecow::from("%di"),
            (DI, 4) => Ecow::from("%edi"),
            (DI, _) => Ecow::from("%rdi"),

            (SI, 1) => Ecow::from("%sil"),
            (SI, 2) => Ecow::from("%si"),
            (SI, 4) => Ecow::from("%esi"),
            (SI, _) => Ecow::from("%rsi"),

            (R8, 1) => Ecow::from("%r8b"),
            (R8, 2) => Ecow::from("%r8w"),
            (R8, 4) => Ecow::from("%r8d"),
            (R8, _) => Ecow::from("%r8"),

            (R9, 1) => Ecow::from("%r9b"),
            (R9, 2) => Ecow::from("%r9w"),
            (R9, 4) => Ecow::from("%r9d"),
            (R9, _) => Ecow::from("%r9"),

            (R10, 1) => Ecow::from("%r10b"),
            (R10, 2) => Ecow::from("%r10w"),
            (R10, 4) => Ecow::from("%r10d"),
            (R10, _) => Ecow::from("%r10"),

            (R11, 1) => Ecow::from("%r11b"),
            (R11, 2) => Ecow::from("%r11w"),
            (R11, 4) => Ecow::from("%r11d"),
            (R11, _) => Ecow::from("%r11"),

            (CX, 1) => Ecow::from("%cl"),
            (CX, 2) => Ecow::from("%cx"),
            (CX, 4) => Ecow::from("%ecx"),
            (CX, _) => Ecow::from("%rcx"),
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
    fn emit_code(self) -> Ecow {
        match self {
            Self::E => Ecow::from("e"),
            Self::NE => Ecow::from("ne"),
            Self::L => Ecow::from("l"),
            Self::LE => Ecow::from("le"),
            Self::G => Ecow::from("g"),
            Self::GE => Ecow::from("ge"),
        }
    }
}

// =========

pub trait ToAsm: std::fmt::Debug + Clone {
    type Output;
    fn to_asm(&self) -> Self::Output;
}
impl ToAsm for () {
    type Output = ();
    fn to_asm(&self) -> Self::Output {}
}
impl<T> ToAsm for Vec<T>
where
    T: ToAsm,
{
    type Output = Vec<T::Output>;
    fn to_asm(&self) -> Self::Output {
        self.iter().map(ToAsm::to_asm).collect()
    }
}

impl ToAsm for ir::Program {
    type Output = Program;
    fn to_asm(&self) -> Self::Output {
        Program { top_level: self.top_level.to_asm(), symbols: self.symbols.clone() }
    }
}

impl ToAsm for ir::TopLevel {
    type Output = TopLevel;
    fn to_asm(&self) -> Self::Output {
        match self {
            ir::TopLevel::Function { name, global, params, body } => {
                let instrs = ARG_REGISTERS
                    .into_iter()
                    .map(|r| Reg(r, 4))
                    .chain((16..).step_by(8).map(Stack))
                    .zip(params.clone())
                    .map(|(src, param)| Instr::Mov(src, Pseudo(param)))
                    .chain(body.iter().flat_map(ToAsm::to_asm))
                    .collect();

                TopLevel::Function { name: name.clone(), instrs, stack_size: 0, global: *global }
            }
            ir::TopLevel::StaticVar { name, global, init } => {
                TopLevel::StaticVariable { name: name.clone(), global: *global, init: *init }
            }
        }
    }
}

impl ToAsm for ir::Instr {
    type Output = Vec<Instr>;
    #[allow(clippy::too_many_lines)]
    fn to_asm(&self) -> Self::Output {
        match self {
            Self::Return(value) => {
                vec![Instr::Mov(value.to_asm(), Operand::Reg(Register::AX, 4)), Instr::Ret]
            }
            Self::Unary { op: ir::UnOp::Not, src, dst } => {
                let dst = dst.to_asm();
                vec![
                    Instr::Cmp(Operand::Imm(0), src.to_asm()),
                    Instr::Mov(Operand::Imm(0), dst.clone()),
                    Instr::SetCC(E, dst),
                ]
            }
            Self::Unary { op, src, dst } => {
                let dst = dst.to_asm();
                vec![Instr::Mov(src.to_asm(), dst.clone()), Instr::Unary(op.to_asm(), dst)]
            }
            Self::Binary { op, lhs, rhs, dst } => match op {
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
                        Instr::Mov(lhs.to_asm(), dst.clone()),
                        Instr::Binary(op.to_asm().unwrap_left(), rhs.to_asm(), dst),
                    ]
                }
                ir::BinOp::Divide | ir::BinOp::Reminder => {
                    let res = match op {
                        ir::BinOp::Divide => Register::AX,
                        ir::BinOp::Reminder => Register::DX,
                        _ => unreachable!(),
                    };
                    vec![
                        Instr::Mov(lhs.to_asm(), Operand::Reg(Register::AX, 4)),
                        Instr::Cdq,
                        Instr::Idiv(rhs.to_asm()),
                        Instr::Mov(Operand::Reg(res, 4), dst.to_asm()),
                    ]
                }
                ir::BinOp::Equal
                | ir::BinOp::NotEqual
                | ir::BinOp::LessThan
                | ir::BinOp::LessOrEqual
                | ir::BinOp::GreaterThan
                | ir::BinOp::GreaterOrEqual => {
                    let dst = dst.to_asm();
                    vec![
                        Instr::Cmp(rhs.to_asm(), lhs.to_asm()),
                        Instr::Mov(Operand::Imm(0), dst.clone()),
                        Instr::SetCC(op.to_asm().unwrap_right(), dst),
                    ]
                }
            },
            Self::Copy { src, dst } => vec![Instr::Mov(src.to_asm(), dst.to_asm())],
            Self::Jump { target } => vec![Instr::Jmp(target.clone())],
            Self::JumpIfZero { cond, target } => {
                vec![Instr::Cmp(Operand::Imm(0), cond.to_asm()), Instr::JmpCC(E, target.clone())]
            }
            Self::JumpIfNotZero { cond, target } => {
                vec![Instr::Cmp(Operand::Imm(0), cond.to_asm()), Instr::JmpCC(NE, target.clone())]
            }
            Self::Label(name) => vec![Instr::Label(name.clone())],
            Self::FuncCall { name, args, dst } => {
                let (reg_args, stack_args) =
                    args.split_at_checked(6).unwrap_or((args.as_slice(), &[]));

                (stack_args.len() % 2 != 0)
                    .then_some(Instr::AllocateStack(8))
                    .into_iter()
                    .chain(
                        ARG_REGISTERS
                            .into_iter()
                            .zip(reg_args)
                            .map(|(reg, arg)| Instr::Mov(arg.to_asm(), Reg(reg, 4))),
                    )
                    .chain(stack_args.iter().rev().flat_map(|arg| match arg.to_asm() {
                        opp @ Imm(_) => vec![Instr::Push(opp)],
                        Reg(r, 4) => vec![Instr::Push(Reg(r, 8))],
                        opp => vec![Instr::Mov(opp, Reg(AX, 4)), Instr::Push(Reg(AX, 8))],
                    }))
                    .chain([Instr::Call(name.clone())])
                    .chain((!stack_args.is_empty()).then_some(Instr::DeallocateStack(
                        8 * stack_args.len() as u32
                            + (stack_args.len() % 2 != 0).then_some(8).unwrap_or_default(),
                    )))
                    .chain([Instr::Mov(Reg(AX, 4), dst.to_asm())])
                    .collect()
            }
        }
    }
}
impl ToAsm for ir::Value {
    type Output = Operand;
    fn to_asm(&self) -> Self::Output {
        match self {
            Self::Const(i) => Operand::Imm(*i),
            Self::Var(place) => place.to_asm(),
        }
    }
}
impl ToAsm for ir::Place {
    type Output = Operand;
    fn to_asm(&self) -> Self::Output {
        Operand::Pseudo(self.0.clone())
    }
}
impl ToAsm for ir::UnOp {
    type Output = Operator;
    fn to_asm(&self) -> Self::Output {
        match self {
            Self::Complement => Operator::Not,
            Self::Negate => Operator::Neg,
            Self::Not => unreachable!("Not is implemented otherwise"),
        }
    }
}
impl ToAsm for ir::BinOp {
    type Output = Either<Operator, CondCode>;
    fn to_asm(&self) -> Self::Output {
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

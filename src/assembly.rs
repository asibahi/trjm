#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

use crate::ir;
use ecow::EcoString;
use either::Either::{self, Left, Right};
use rustc_hash::FxHashMap;
use std::{collections::hash_map::Entry, io::Write};

pub trait Assembly: Clone {
    fn emit_code(&self, f: &mut impl Write);

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, i32>, stack_depth: &mut i32);
    fn adjust_instrs(&mut self);
}
impl Assembly for () {
    fn emit_code(&self, _: &mut impl Write) {}
    fn replace_pseudos(&mut self, _: &mut FxHashMap<EcoString, i32>, _: &mut i32) {}
    fn adjust_instrs(&mut self) {}
}
impl<T> Assembly for Vec<T>
where
    T: Assembly,
{
    fn emit_code(&self, f: &mut impl Write) {
        self.iter().for_each(|e| e.emit_code(f));
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, i32>, stack_depth: &mut i32) {
        self.iter_mut().for_each(|e| e.replace_pseudos(map, stack_depth));
    }

    fn adjust_instrs(&mut self) {
        self.iter_mut().for_each(Assembly::adjust_instrs);
    }
}
impl<T, U> Assembly for Either<T, U>
where
    T: Assembly,
    U: Assembly,
{
    fn emit_code(&self, f: &mut impl Write) {
        self.as_ref().either_with(f, |f, t| t.emit_code(f), |f, u| u.emit_code(f));
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, i32>, stack_depth: &mut i32) {
        self.as_mut().either_with(
            (map, stack_depth),
            |(m, s), t| t.replace_pseudos(m, s),
            |(m, s), u| u.replace_pseudos(m, s),
        );
    }

    fn adjust_instrs(&mut self) {
        self.as_mut().either(Assembly::adjust_instrs, Assembly::adjust_instrs);
    }
}

macro_rules! to_str {
    ($item:expr) => {{
        let mut buf = ecow::EcoVec::new();
        $item.emit_code(&mut buf);
        EcoString::from(String::from_utf8_lossy(&buf))
    }};
}

#[derive(Debug, Clone)]
pub struct Program(pub Vec<FuncDef>);
impl Program {
    pub fn fixup_passes(&mut self) {
        let mut map = FxHashMap::<EcoString, i32>::default();
        let mut stack_depth = 0;

        self.replace_pseudos(&mut map, &mut stack_depth);
        self.adjust_instrs();
    }
}
impl Assembly for Program {
    fn emit_code(&self, f: &mut impl Write) {
        self.0.emit_code(f);
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, i32>, stack_depth: &mut i32) {
        self.0.replace_pseudos(map, stack_depth);
    }

    fn adjust_instrs(&mut self) {
        self.0.adjust_instrs();
    }
}

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: EcoString,
    pub instrs: Vec<Instr>,
    pub stack_size: i32,
}
impl Assembly for FuncDef {
    fn emit_code(&self, f: &mut impl Write) {
        _ = writeln!(f, "\t.globl  _{}", self.name);
        _ = writeln!(f, "_{}:", self.name);

        _ = writeln!(f, "\tpushq   %rbp");
        _ = writeln!(f, "\tmovq    %rsp,   %rbp");

        self.instrs.emit_code(f);
    }

    fn replace_pseudos(&mut self, _: &mut FxHashMap<EcoString, i32>, _: &mut i32) {
        let mut map = FxHashMap::<EcoString, i32>::default();
        let mut stack_depth = 0;

        self.instrs.replace_pseudos(&mut map, &mut stack_depth);

        self.stack_size = stack_depth;
    }

    fn adjust_instrs(&mut self) {
        '_stack_frame: {
            let sd = self.stack_size.unsigned_abs().next_multiple_of(16);

            self.instrs.insert(0, Instr::AllocateStack(sd));
        }

        '_fixup_instrs: for _ in 0..2 {
            let mut out = Vec::with_capacity(self.instrs.len());

            for instr in std::mem::take(&mut self.instrs) {
                match instr {
                    Instr::SetCC(cc, Reg(r, 1)) => {
                        out.push(Instr::SetCC(cc, Reg(r, 1)));
                    }
                    Instr::Mov(Imm(0), dst @ Reg(..)) => {
                        out.push(Instr::Binary(Operator::Xor, dst.clone(), dst));
                    }
                    Instr::Mov(src @ Stack(_), dst @ Stack(_)) => {
                        out.extend([Instr::Mov(src, Reg(R10, 4)), Instr::Mov(Reg(R10, 4), dst)]);
                    }
                    Instr::Cmp(src @ Stack(_), dst @ Stack(_)) => {
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
                        src @ Stack(_),
                        dst @ Stack(_),
                    ) => out.extend([
                        Instr::Mov(src, Reg(R10, 4)),
                        Instr::Binary(opp, Reg(R10, 4), dst),
                    ]),
                    Instr::Binary(Operator::Mul, src, dst @ Stack(_)) => out.extend([
                        Instr::Mov(dst.clone(), Reg(R11, 4)),
                        Instr::Binary(Operator::Mul, src, Reg(R11, 4)),
                        Instr::Mov(Reg(R11, 4), dst),
                    ]),
                    Instr::Binary(opp @ (Operator::Shl | Operator::Shr), src, dst @ Stack(_))
                        if src != Reg(CX, 1) =>
                    {
                        out.extend([
                            Instr::Mov(src, Reg(CX, 4)),
                            Instr::Binary(opp, Reg(CX, 1), dst),
                        ]);
                    }
                    other => out.push(other),
                }
            }
            self.instrs = out;
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
    Jmp(EcoString),
    JmpCC(CondCode, EcoString),
    SetCC(CondCode, Operand),
    Label(EcoString),
    AllocateStack(u32),
    DeallocateStack(u32),
    Push(Operand),
    Call(EcoString),
    Ret,
}
impl Assembly for Instr {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Self::Mov(src, dst) => {
                let src = to_str!(src) + ",";
                let dst = to_str!(dst);

                _ = writeln!(f, "\tmovl    {src:<7} {dst}");
            }
            Self::Unary(un_op, operand) => {
                let uo = to_str!(un_op);
                let op = to_str!(operand);

                _ = writeln!(f, "\t{uo:<7} {op}");
            }
            Self::Binary(bin_op, op1, op2) => {
                let bo = to_str!(bin_op);
                let o1 = to_str!(op1) + ",";
                let o2 = to_str!(op2);

                _ = writeln!(f, "\t{bo:<7} {o1:<7} {o2}");
            }
            Self::Idiv(op) => _ = writeln!(f, "\tidivl    {}", to_str!(op)),

            Self::Cdq => _ = writeln!(f, "\tcdq"),

            Self::AllocateStack(i) => {
                _ = writeln!(f, "\tsubq    {:<7} %rsp", to_str!(Imm(*i as i32)) + ",");
            }
            Self::DeallocateStack(i) => {
                _ = writeln!(f, "\taddq    {:<7} %rsp", to_str!(Imm(*i as i32)) + ",");
            }

            Self::Push(op) => _ = writeln!(f, "\tpushq   {}", to_str!(op)),
            Self::Call(label) => _ = writeln!(f, "\tcall    _{label}"),

            Self::Ret => {
                _ = writeln!(f, "\tmovq    %rbp,   %rsp");
                _ = writeln!(f, "\tpopq    %rbp");
                _ = writeln!(f, "\tret");
            }
            Self::Cmp(op1, op2) => {
                let op1 = to_str!(op1) + ",";
                let op2 = to_str!(op2);

                _ = writeln!(f, "\tcmpl    {op1:<7} {op2}");
            }
            Self::Jmp(label) => _ = writeln!(f, "\tjmp     .L{label}"),
            Self::JmpCC(cond, label) => {
                let cond = to_str!(cond);
                _ = writeln!(f, "\tj{cond:<6} .L{label}");
            }
            Self::SetCC(cond, op) => {
                let cond = to_str!(cond);
                let op = to_str!(op);

                _ = writeln!(f, "\tset{cond:<4} {op}");
            }
            Self::Label(label) => _ = writeln!(f, ".L{label}:"),
        }
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, i32>, stack_depth: &mut i32) {
        match self {
            Self::Unary(_, opp) | Self::Idiv(opp) | Self::SetCC(_, opp) => {
                opp.replace_pseudos(map, stack_depth);
            }
            Self::Binary(_, opp1, opp2) | Self::Mov(opp1, opp2) | Self::Cmp(opp1, opp2) => {
                opp1.replace_pseudos(map, stack_depth);
                opp2.replace_pseudos(map, stack_depth);
            }
            Self::Push(opp) => opp.replace_pseudos(map, stack_depth),

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

    fn adjust_instrs(&mut self) {}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    Imm(i32),
    Reg(Register, usize),
    Pseudo(EcoString),
    Stack(i32),
}
use Operand::*;
impl Assembly for Operand {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Imm(i) => _ = write!(f, "${i}"),
            Reg(r, s) => r.emit_code(*s, f),
            Stack(i) => _ = write!(f, "{i}(%rbp)"),
            Pseudo(e) => unreachable!("pseudo register {e} printed"),
        }
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, i32>, stack_depth: &mut i32) {
        let Pseudo(name) = self else {
            return;
        };

        let stack = if let Entry::Vacant(e) = map.entry(name.clone()) {
            *stack_depth -= 4; // 4 bytes
            e.insert(*stack_depth);
            *stack_depth
        } else {
            map[name]
        };

        std::mem::swap(self, &mut Stack(stack));
    }
    fn adjust_instrs(&mut self) {}
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
impl Assembly for Operator {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Self::Not => _ = write!(f, "notl"),
            Self::Neg => _ = write!(f, "negl"),
            Self::Add => _ = write!(f, "addl"),
            Self::Sub => _ = write!(f, "subl"),
            Self::Mul => _ = write!(f, "imull"),
            Self::And => _ = write!(f, "andl"),
            Self::Or => _ = write!(f, "orl"),
            Self::Xor => _ = write!(f, "xorl"),
            Self::Shr => _ = write!(f, "sarl"),
            Self::Shl => _ = write!(f, "shll"),
        }
    }

    fn replace_pseudos(&mut self, _: &mut FxHashMap<EcoString, i32>, _: &mut i32) {}
    fn adjust_instrs(&mut self) {}
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
    fn emit_code(self, byte_size: usize, f: &mut impl Write) {
        assert!(byte_size == 1 || byte_size == 2 || byte_size == 4 || byte_size == 8);
        match (self, byte_size) {
            (AX, 1) => _ = write!(f, "%al"),
            (AX, 2) => _ = write!(f, "%ax"),
            (AX, 4) => _ = write!(f, "%eax"),
            (AX, _) => _ = write!(f, "%rax"),

            (DX, 1) => _ = write!(f, "%dl"),
            (DX, 2) => _ = write!(f, "%dx"),
            (DX, 4) => _ = write!(f, "%edx"),
            (DX, _) => _ = write!(f, "%rdx"),

            (DI, 1) => _ = write!(f, "%dil"),
            (DI, 2) => _ = write!(f, "%di"),
            (DI, 4) => _ = write!(f, "%edi"),
            (DI, _) => _ = write!(f, "%rdi"),

            (SI, 1) => _ = write!(f, "%sil"),
            (SI, 2) => _ = write!(f, "%si"),
            (SI, 4) => _ = write!(f, "%esi"),
            (SI, _) => _ = write!(f, "%rsi"),

            (R8, 1) => _ = write!(f, "%r8b"),
            (R8, 2) => _ = write!(f, "%r8w"),
            (R8, 4) => _ = write!(f, "%r8d"),
            (R8, _) => _ = write!(f, "%r8"),

            (R9, 1) => _ = write!(f, "%r9b"),
            (R9, 2) => _ = write!(f, "%r9w"),
            (R9, 4) => _ = write!(f, "%r9d"),
            (R9, _) => _ = write!(f, "%r9"),

            (R10, 1) => _ = write!(f, "%r10b"),
            (R10, 2) => _ = write!(f, "%r10w"),
            (R10, 4) => _ = write!(f, "%r10d"),
            (R10, _) => _ = write!(f, "%r10"),

            (R11, 1) => _ = write!(f, "%r11b"),
            (R11, 2) => _ = write!(f, "%r11w"),
            (R11, 4) => _ = write!(f, "%r11d"),
            (R11, _) => _ = write!(f, "%r11"),

            (CX, 1) => _ = write!(f, "%cl"),
            (CX, 2) => _ = write!(f, "%cx"),
            (CX, 4) => _ = write!(f, "%ecx"),
            (CX, _) => _ = write!(f, "%rcx"),
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
impl Assembly for CondCode {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Self::E => _ = write!(f, "e"),
            Self::NE => _ = write!(f, "ne"),
            Self::L => _ = write!(f, "l"),
            Self::LE => _ = write!(f, "le"),
            Self::G => _ = write!(f, "g"),
            Self::GE => _ = write!(f, "ge"),
        }
    }

    fn replace_pseudos(&mut self, _: &mut FxHashMap<EcoString, i32>, _: &mut i32) {}
    fn adjust_instrs(&mut self) {}
}

// =========

pub trait ToAsm: std::fmt::Debug + Clone {
    type Output: Assembly;
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
        Program(self.0.to_asm())
    }
}

impl ToAsm for ir::TopLevel {
    type Output = FuncDef;
    fn to_asm(&self) -> Self::Output {
        // let instrs = ARG_REGISTERS
        //     .into_iter()
        //     .map(|r| Reg(r, 4))
        //     .chain((16..).step_by(8).map(Stack))
        //     .zip(self.params.clone())
        //     .map(|(src, param)| Instr::Mov(src, Pseudo(param)))
        //     .chain(self.body.iter().flat_map(ToAsm::to_asm))
        //     .collect();

        // FuncDef { name: self.name.clone(), instrs, stack_size: 0 }

        todo!()
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

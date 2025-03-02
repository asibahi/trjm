use crate::ir;
use ecow::EcoString;
use either::Either::{self, Left, Right};
use rustc_hash::FxHashMap;
use std::{collections::hash_map::Entry, io::Write};

pub trait Assembly: Clone {
    fn emit_code(&self, f: &mut impl Write);

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, u32>, stack_depth: &mut u32);
    fn adjust_instrs(&mut self, stack_depth: u32);
}
impl Assembly for () {
    fn emit_code(&self, _: &mut impl Write) {}
    fn replace_pseudos(&mut self, _: &mut FxHashMap<EcoString, u32>, _: &mut u32) {}
    fn adjust_instrs(&mut self, _: u32) {}
}
impl<T> Assembly for Vec<T>
where
    T: Assembly,
{
    fn emit_code(&self, f: &mut impl Write) {
        self.iter().for_each(|e| e.emit_code(f));
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, u32>, stack_depth: &mut u32) {
        self.iter_mut().for_each(|e| e.replace_pseudos(map, stack_depth));
    }

    fn adjust_instrs(&mut self, stack_depth: u32) {
        self.iter_mut().for_each(|e| e.adjust_instrs(stack_depth));
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

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, u32>, stack_depth: &mut u32) {
        self.as_mut().either_with(
            (map, stack_depth),
            |(m, s), t| t.replace_pseudos(m, s),
            |(m, s), u| u.replace_pseudos(m, s),
        );
    }

    fn adjust_instrs(&mut self, stack_depth: u32) {
        self.as_mut().either(|t| t.adjust_instrs(stack_depth), |u| u.adjust_instrs(stack_depth));
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
pub struct Program(pub FuncDef);
impl Program {
    pub fn fixup_passes(&mut self) {
        let mut map = FxHashMap::<EcoString, u32>::default();
        let mut stack_depth = 0;

        self.replace_pseudos(&mut map, &mut stack_depth);
        self.adjust_instrs(stack_depth);
    }
}
impl Assembly for Program {
    fn emit_code(&self, f: &mut impl Write) {
        self.0.emit_code(f);
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, u32>, stack_depth: &mut u32) {
        self.0.replace_pseudos(map, stack_depth);
    }

    fn adjust_instrs(&mut self, stack_depth: u32) {
        self.0.adjust_instrs(stack_depth);
    }
}

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: EcoString,
    pub instrs: Vec<Instr>,
}
impl Assembly for FuncDef {
    fn emit_code(&self, f: &mut impl Write) {
        _ = writeln!(f, "\t.globl  _{}", self.name);
        _ = writeln!(f, "_{}:", self.name);

        _ = writeln!(f, "\tpushq   %rbp");
        _ = writeln!(f, "\tmovq    %rsp,   %rbp");

        self.instrs.emit_code(f);
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, u32>, stack_depth: &mut u32) {
        self.instrs.replace_pseudos(map, stack_depth);
    }

    fn adjust_instrs(&mut self, stack_depth: u32) {
        '_stack_frame: {
            self.instrs.insert(0, Instr::AllocateStack(stack_depth));
        }

        '_fixup_instrs: for _ in 0..2 {
            let mut out = Vec::with_capacity(self.instrs.len());

            for instr in std::mem::take(&mut self.instrs) {
                match instr {
                    Instr::SetCC(cc, Reg(r)) => {
                        out.push(Instr::SetCC(cc, Reg(r.as_1_byte())));
                    }
                    Instr::Mov(Imm(0), dst @ Reg(_)) => {
                        out.push(Instr::Binary(Operator::Xor, dst.clone(), dst));
                    }
                    Instr::Mov(src @ Stack(_), dst @ Stack(_)) => {
                        out.extend([Instr::Mov(src, Reg(R10)), Instr::Mov(Reg(R10), dst)]);
                    }
                    Instr::Cmp(src @ Stack(_), dst @ Stack(_)) => {
                        out.extend([Instr::Mov(src, Reg(R10)), Instr::Cmp(Reg(R10), dst)]);
                    }
                    Instr::Cmp(src, dst @ Imm(_)) => {
                        out.extend([Instr::Mov(dst, Reg(R11)), Instr::Cmp(src, Reg(R11))]);
                    }
                    Instr::Idiv(v @ Imm(_)) => {
                        out.extend([Instr::Mov(v, Reg(R10)), Instr::Idiv(Reg(R10))]);
                    }
                    Instr::Binary(
                        opp @ (Operator::Add
                        | Operator::Sub
                        | Operator::And
                        | Operator::Or
                        | Operator::Xor),
                        src @ Stack(_),
                        dst @ Stack(_),
                    ) => out.extend([Instr::Mov(src, Reg(R10)), Instr::Binary(opp, Reg(R10), dst)]),
                    Instr::Binary(Operator::Mul, src, dst @ Stack(_)) => out.extend([
                        Instr::Mov(dst.clone(), Reg(R11)),
                        Instr::Binary(Operator::Mul, src, Reg(R11)),
                        Instr::Mov(Reg(R11), dst),
                    ]),
                    Instr::Binary(opp @ (Operator::Shl | Operator::Shr), src, dst @ Stack(_))
                        if src != Reg(CX.as_1_byte()) =>
                    {
                        out.extend([
                            Instr::Mov(src, Reg(CX)),
                            Instr::Binary(opp, Reg(CX.as_1_byte()), dst),
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
            Self::Idiv(operand) => {
                let op = to_str!(operand);
                _ = writeln!(f, "\tidivl    {op}");
            }
            Self::Cdq => _ = writeln!(f, "\tcdq"),
            Self::AllocateStack(i) => {
                _ = {
                    let op = to_str!(Imm(*i)) + ",";
                    writeln!(f, "\tsubq    {op:<7} %rsp")
                }
            }
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

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, u32>, stack_depth: &mut u32) {
        match self {
            Self::Mov(opp1, opp2) | Self::Cmp(opp1, opp2) => {
                opp1.replace_pseudos(map, stack_depth);
                opp2.replace_pseudos(map, stack_depth);
            }
            Self::Unary(_, opp) | Self::Idiv(opp) | Self::SetCC(_, opp) => {
                opp.replace_pseudos(map, stack_depth);
            }
            Self::Binary(bin_op, opp1, opp2) => {
                bin_op.replace_pseudos(map, stack_depth);
                opp1.replace_pseudos(map, stack_depth);
                opp2.replace_pseudos(map, stack_depth);
            }
            Self::Cdq
            | Self::AllocateStack(_)
            | Self::Ret
            | Self::Jmp(..)
            | Self::JmpCC(..)
            | Self::Label(..) => (),
        }
    }

    fn adjust_instrs(&mut self, _: u32) {}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    Imm(u32),
    Reg(Register),
    Pseudo(EcoString),
    Stack(u32),
}
use Operand::*;
impl Assembly for Operand {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Imm(i) => _ = write!(f, "${i}"),
            Reg(r) => r.emit_code(f),
            Stack(i) => _ = write!(f, "-{i}(%rbp)"),
            Pseudo(e) => unreachable!("pseudo register {e} printed"),
        }
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, u32>, stack_depth: &mut u32) {
        let Pseudo(name) = self else {
            return;
        };

        let stack = if let Entry::Vacant(e) = map.entry(name.clone()) {
            *stack_depth += 4; // 4 bytes
            e.insert(*stack_depth);
            *stack_depth
        } else {
            map[name]
        };

        std::mem::swap(self, &mut Stack(stack));
    }
    fn adjust_instrs(&mut self, _: u32) {}
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

    fn replace_pseudos(&mut self, _: &mut FxHashMap<EcoString, u32>, _: &mut u32) {}
    fn adjust_instrs(&mut self, _: u32) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[rustfmt::skip]
pub enum Register {
    AX, AL,
    DX, DL,
    R10, R10B,
    R11, R11B,
    CX, CL,
}
use Register::*;
impl Register {
    #[expect(unused)]
    fn as_4_byte(self) -> Self {
        match self {
            AX | AL => AX,
            DX | DL => DX,
            R10 | R10B => R10,
            R11 | R11B => R11,
            CX | CL => CX,
        }
    }
    fn as_1_byte(self) -> Self {
        match self {
            AX | AL => AL,
            DX | DL => DL,
            R10 | R10B => R10B,
            R11 | R11B => R11B,
            CX | CL => CL,
        }
    }
}
impl Assembly for Register {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            AX => _ = write!(f, "%eax"),
            AL => _ = write!(f, "%al"),
            DX => _ = write!(f, "%edx"),
            DL => _ = write!(f, "%dl"),
            R10 => _ = write!(f, "%r10d"),
            R10B => _ = write!(f, "%r10b"),
            R11 => _ = write!(f, "%r11d"),
            R11B => _ = write!(f, "%r11b"),
            CX => _ = write!(f, "%ecx"),
            CL => _ = write!(f, "%cl"),
        }
    }

    fn replace_pseudos(&mut self, _: &mut FxHashMap<EcoString, u32>, _: &mut u32) {}
    fn adjust_instrs(&mut self, _: u32) {}
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

    fn replace_pseudos(&mut self, _: &mut FxHashMap<EcoString, u32>, _: &mut u32) {}
    fn adjust_instrs(&mut self, _: u32) {}
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
        // Program(self.0.to_asm())
        todo!()
    }
}

impl ToAsm for ir::FuncDef {
    type Output = FuncDef;
    fn to_asm(&self) -> Self::Output {
        FuncDef {
            name: self.name.clone(),
            instrs: self.body.to_asm().iter().flatten().cloned().collect(),
        }
    }
}

impl ToAsm for ir::Instr {
    type Output = Vec<Instr>;
    fn to_asm(&self) -> Self::Output {
        match self {
            Self::Return(value) => {
                vec![Instr::Mov(value.to_asm(), Operand::Reg(Register::AX)), Instr::Ret]
            }
            Self::Unary { op: ir::UnOp::Not, src, dst } => {
                let dst = dst.to_asm();
                vec![
                    Instr::Cmp(Operand::Imm(0), src.to_asm()),
                    Instr::Mov(Operand::Imm(0), dst.clone()), // zero stuff : replacing with XOR breaks code
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
                        Instr::Mov(lhs.to_asm(), Operand::Reg(Register::AX)),
                        Instr::Cdq,
                        Instr::Idiv(rhs.to_asm()),
                        Instr::Mov(Operand::Reg(res), dst.to_asm()),
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
        }
    }
}
impl ToAsm for ir::Value {
    type Output = Operand;
    fn to_asm(&self) -> Self::Output {
        #[expect(clippy::cast_sign_loss)]
        match self {
            Self::Const(i) => Operand::Imm(*i as u32),
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

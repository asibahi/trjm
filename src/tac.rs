#![allow(refining_impl_trait_internal)]

use crate::assembly::{
    self,
    CondCode::{E, NE},
    Instr as AsmInst, Operand, Operator, Register,
};
use ecow::EcoString;
use either::Either::{self, Left, Right};

pub trait Tac: std::fmt::Debug + Clone {
    type Output: assembly::Assembly;
    fn to_asm(&self) -> Self::Output;
}
impl<T> Tac for Vec<T>
where
    T: Tac,
{
    type Output = Vec<T::Output>;
    fn to_asm(&self) -> Self::Output {
        self.iter().map(Tac::to_asm).collect()
    }
}

#[derive(Debug, Clone)]
pub struct Program(pub FuncDef);
impl Tac for Program {
    type Output = assembly::Program;
    fn to_asm(&self) -> Self::Output {
        assembly::Program(self.0.to_asm())
    }
}

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: EcoString,
    pub body: Vec<Instr>,
}
impl Tac for FuncDef {
    type Output = assembly::FuncDef;
    fn to_asm(&self) -> Self::Output {
        assembly::FuncDef {
            name: self.name.clone(),
            instrs: self.body.to_asm().iter().flatten().cloned().collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Instr {
    Return(Value),
    Unary {
        op: UnOp,
        src: Value,
        dst: Place,
    },
    Binary {
        op: BinOp,
        src1: Value,
        src2: Value,
        dst: Place,
    },
    Copy {
        src: Value,
        dst: Place,
    },
    Jump {
        target: EcoString,
    },
    JumpIfZero {
        cond: Value,
        target: EcoString,
    },
    JumpIfNotZero {
        cond: Value,
        target: EcoString,
    },
    Label(EcoString),
}
impl Tac for Instr {
    type Output = Vec<AsmInst>;
    fn to_asm(&self) -> Self::Output {
        match self {
            Instr::Return(value) => vec![
                AsmInst::Mov(value.to_asm(), Operand::Reg(Register::AX)),
                AsmInst::Ret,
            ],
            Instr::Unary {
                op: UnOp::Not,
                src,
                dst,
            } => {
                let dst = dst.to_asm();
                vec![
                    AsmInst::Cmp(Operand::Imm(0), src.to_asm()),
                    AsmInst::Mov(Operand::Imm(0), dst.clone()), // zero stuff : replacing with XOR breaks code
                    AsmInst::SetCC(E, dst),
                ]
            }
            Instr::Unary { op, src, dst } => {
                let dst = dst.to_asm();
                vec![
                    AsmInst::Mov(src.to_asm(), dst.clone()),
                    AsmInst::Unary(op.to_asm(), dst),
                ]
            }
            Instr::Binary {
                op,
                src1,
                src2,
                dst,
            } => match op {
                BinOp::Add
                | BinOp::Subtract
                | BinOp::Multiply
                | BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::BitXor
                | BinOp::LeftShift
                | BinOp::RightShift => {
                    let dst = dst.to_asm();
                    vec![
                        AsmInst::Mov(src1.to_asm(), dst.clone()),
                        AsmInst::Binary(op.to_asm().unwrap_left(), src2.to_asm(), dst),
                    ]
                }
                BinOp::Divide | BinOp::Reminder => {
                    let res = match op {
                        BinOp::Divide => Register::AX,
                        BinOp::Reminder => Register::DX,
                        _ => unreachable!(),
                    };
                    vec![
                        AsmInst::Mov(src1.to_asm(), Operand::Reg(Register::AX)),
                        AsmInst::Cdq,
                        AsmInst::Idiv(src2.to_asm()),
                        AsmInst::Mov(Operand::Reg(res), dst.to_asm()),
                    ]
                }
                BinOp::Equal
                | BinOp::NotEqual
                | BinOp::LessThan
                | BinOp::LessOrEqual
                | BinOp::GreaterThan
                | BinOp::GreaterOrEqual => {
                    let dst = dst.to_asm();
                    vec![
                        AsmInst::Cmp(src2.to_asm(), src1.to_asm()),
                        AsmInst::Mov(Operand::Imm(0), dst.clone()), // zero stuff : replacing with XOR breaks code
                        AsmInst::SetCC(op.to_asm().unwrap_right(), dst),
                    ]
                }
            },
            Instr::Copy { src, dst } => vec![AsmInst::Mov(src.to_asm(), dst.to_asm())],
            Instr::Jump { target } => vec![AsmInst::Jmp(target.clone())],
            Instr::JumpIfZero { cond, target } => vec![
                AsmInst::Cmp(Operand::Imm(0), cond.to_asm()),
                AsmInst::JmpCC(E, target.clone()),
            ],
            Instr::JumpIfNotZero { cond, target } => vec![
                AsmInst::Cmp(Operand::Imm(0), cond.to_asm()),
                AsmInst::JmpCC(NE, target.clone()),
            ],
            Instr::Label(name) => vec![AsmInst::Label(name.clone())],
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Const(u32),
    Var(Place),
}
impl Tac for Value {
    type Output = Operand;
    fn to_asm(&self) -> Self::Output {
        match self {
            Value::Const(i) => Operand::Imm(*i),
            Value::Var(place) => place.to_asm(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Place(pub EcoString);
impl Tac for Place {
    type Output = Operand;
    fn to_asm(&self) -> Self::Output {
        Operand::Pseudo(self.0.clone())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UnOp {
    Complement,
    Negate,
    Not,
}
impl Tac for UnOp {
    type Output = Operator;
    fn to_asm(&self) -> Self::Output {
        match self {
            UnOp::Complement => Operator::Not,
            UnOp::Negate => Operator::Neg,
            UnOp::Not => unreachable!("Not is implemented otherwise"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Reminder,
    // Chapter 3 Extra credit
    BitAnd,
    BitOr,
    BitXor,
    LeftShift,
    RightShift,

    // Chapter 4
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
}
impl Tac for BinOp {
    type Output = Either<Operator, assembly::CondCode>;
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

            BinOp::Equal => Right(assembly::CondCode::E),
            BinOp::NotEqual => Right(assembly::CondCode::NE),
            BinOp::LessThan => Right(assembly::CondCode::L),
            BinOp::LessOrEqual => Right(assembly::CondCode::LE),
            BinOp::GreaterThan => Right(assembly::CondCode::G),
            BinOp::GreaterOrEqual => Right(assembly::CondCode::GE),
        }
    }
}

#![allow(refining_impl_trait_internal)]

use crate::assembly;
use ecow::EcoString;

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
    type Output = Vec<assembly::Instr>;
    fn to_asm(&self) -> Self::Output {
        match self {
            Instr::Return(value) => vec![
                assembly::Instr::Mov(
                    value.to_asm(),
                    assembly::Operand::Reg(assembly::Register::AX),
                ),
                assembly::Instr::Ret,
            ],
            Instr::Unary { op, src, dst } => {
                let dst = dst.to_asm();
                vec![
                    assembly::Instr::Mov(src.to_asm(), dst.clone()),
                    assembly::Instr::Unary(op.to_asm(), dst),
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
                        assembly::Instr::Mov(src1.to_asm(), dst.clone()),
                        assembly::Instr::Binary(op.to_asm(), src2.to_asm(), dst),
                    ]
                }
                BinOp::Divide | BinOp::Reminder => {
                    let res = match op {
                        BinOp::Divide => assembly::Register::AX,
                        BinOp::Reminder => assembly::Register::DX,
                        _ => unreachable!(),
                    };
                    vec![
                        assembly::Instr::Mov(
                            src1.to_asm(),
                            assembly::Operand::Reg(assembly::Register::AX),
                        ),
                        assembly::Instr::Cdq,
                        assembly::Instr::Idiv(src2.to_asm()),
                        assembly::Instr::Mov(assembly::Operand::Reg(res), dst.to_asm()),
                    ]
                }
                BinOp::Equal => todo!(),
                BinOp::NotEqual => todo!(),
                BinOp::LessThan => todo!(),
                BinOp::LessOrEqual => todo!(),
                BinOp::GreaterThan => todo!(),
                BinOp::GreaterOrEqual => todo!(),
            },
            Instr::Copy { .. } => todo!(),
            Instr::Jump { .. } => todo!(),
            Instr::JumpIfZero { .. } => todo!(),
            Instr::JumpIfNotZero { .. } => todo!(),
            Instr::Label(_) => todo!(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Const(u32),
    Var(Place),
}
impl Tac for Value {
    type Output = assembly::Operand;
    fn to_asm(&self) -> Self::Output {
        match self {
            Value::Const(i) => assembly::Operand::Imm(*i),
            Value::Var(place) => place.to_asm(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Place(pub EcoString);
impl Tac for Place {
    type Output = assembly::Operand;
    fn to_asm(&self) -> Self::Output {
        assembly::Operand::Pseudo(self.0.clone())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UnOp {
    Complement,
    Negate,
    Not,
}
impl Tac for UnOp {
    type Output = assembly::Operator;
    fn to_asm(&self) -> Self::Output {
        match self {
            UnOp::Complement => assembly::Operator::Not,
            UnOp::Negate => assembly::Operator::Neg,
            UnOp::Not => todo!(),
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
    type Output = assembly::Operator;
    fn to_asm(&self) -> Self::Output {
        match self {
            Self::Add => assembly::Operator::Add,
            Self::Subtract => assembly::Operator::Sub,
            Self::Multiply => assembly::Operator::Mul,
            Self::BitAnd => assembly::Operator::And,
            Self::BitOr => assembly::Operator::Or,
            Self::BitXor => assembly::Operator::Xor,
            Self::LeftShift => assembly::Operator::Shl,
            Self::RightShift => assembly::Operator::Shr,

            Self::Divide | Self::Reminder => {
                unreachable!("Divide and Reminder are implemented in other ways")
            }

            BinOp::Equal => todo!(),
            BinOp::NotEqual => todo!(),
            BinOp::LessThan => todo!(),
            BinOp::LessOrEqual => todo!(),
            BinOp::GreaterThan => todo!(),
            BinOp::GreaterOrEqual => todo!(),
        }
    }
}

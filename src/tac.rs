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
    Unary { op: UnOp, src: Value, dst: Place },
}
impl Tac for Instr {
    type Output = Vec<assembly::Instr>;
    fn to_asm(&self) -> Self::Output {
        match self {
            Instr::Return(value) => vec![
                assembly::Instr::Mov {
                    src: value.to_asm(),
                    dst: assembly::Operand::Reg(assembly::Register::AX),
                },
                assembly::Instr::Ret,
            ],
            Instr::Unary { op, src, dst } => {
                let dst = dst.to_asm();
                vec![
                    assembly::Instr::Mov {
                        src: src.to_asm(),
                        dst: dst.clone(),
                    },
                    assembly::Instr::Unary(op.to_asm(), dst),
                ]
            }
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

#[derive(Debug, Clone)]
pub enum UnOp {
    Complement,
    Negate,
}
impl Tac for UnOp {
    type Output = assembly::UnOp;
    fn to_asm(&self) -> Self::Output {
        match self {
            UnOp::Complement => assembly::UnOp::Not,
            UnOp::Negate => assembly::UnOp::Neg,
        }
    }
}

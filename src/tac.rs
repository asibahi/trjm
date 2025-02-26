#![allow(refining_impl_trait_internal)]

use crate::asm;
use ecow::EcoString;

pub trait Tac: std::fmt::Debug + Clone {
    type Output: asm::Assembly;
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
    type Output = asm::Program;
    fn to_asm(&self) -> Self::Output {
        asm::Program(self.0.to_asm())
    }
}

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: EcoString,
    pub body: Vec<Instr>,
}
impl Tac for FuncDef {
    type Output = asm::FuncDef;
    fn to_asm(&self) -> Self::Output {
        asm::FuncDef {
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
    type Output = Vec<asm::Instr>;
    fn to_asm(&self) -> Self::Output {
        match self {
            Instr::Return(value) => vec![
                asm::Instr::Mov {
                    src: value.to_asm(),
                    dst: asm::Operand::Reg(asm::Register::AX),
                },
                asm::Instr::Ret,
            ],
            Instr::Unary { op, src, dst } => {
                let dst = dst.to_asm();
                vec![
                    asm::Instr::Mov {
                        src: src.to_asm(),
                        dst: dst.clone(),
                    },
                    asm::Instr::Unary(op.to_asm(), dst),
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
    type Output = asm::Operand;
    fn to_asm(&self) -> Self::Output {
        match self {
            Value::Const(i) => asm::Operand::Imm(*i),
            Value::Var(place) => place.to_asm(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Place(pub EcoString);
impl Tac for Place {
    type Output = asm::Operand;
    fn to_asm(&self) -> Self::Output {
        asm::Operand::Pseudo(self.0.clone())
    }
}

#[derive(Debug, Clone)]
pub enum UnOp {
    Complement,
    Negate,
}
impl Tac for UnOp {
    type Output = asm::UnOp;
    fn to_asm(&self) -> Self::Output {
        match self {
            UnOp::Complement => asm::UnOp::Not,
            UnOp::Negate => asm::UnOp::Neg,
        }
    }
}

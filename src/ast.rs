#![allow(refining_impl_trait_internal)]

use crate::asm;
use ecow::EcoString;

pub trait Node {
    fn to_asm(&self) -> impl asm::Asm;
}

#[derive(Debug)]
pub struct Program(pub FuncDef);
impl Node for Program {
    fn to_asm(&self) -> asm::Program {
        asm::Program(self.0.to_asm())
    }
}

#[derive(Debug)]
pub struct FuncDef {
    pub name: Ident,
    pub body: Stmt,
}
impl Node for FuncDef {
    fn to_asm(&self) -> asm::FuncDef {
        asm::FuncDef {
            name: self.name.to_asm(),
            instrs: self.body.to_asm(),
        }
    }
}

#[derive(Debug)]
pub struct Ident(pub EcoString);
impl Node for Ident {
    fn to_asm(&self) -> asm::Ident {
        asm::Ident(self.0.clone())
    }
}

#[derive(Debug)]
pub enum Stmt {
    Return(Box<Expr>),
    If {
        cond: Box<Expr>,
        then: Box<Stmt>,
        else_: Option<Box<Stmt>>,
    },
}
impl Node for Stmt {
    fn to_asm(&self) -> Vec<asm::Instr> {
        match self {
            Stmt::Return(expr) => {
                vec![
                    asm::Instr::Mov {
                        src: expr.to_asm(),
                        dst: asm::Operand::Register,
                    },
                    asm::Instr::Ret,
                ]
            }
            Stmt::If { cond, then, else_ } => todo!(),
        }
    }
}

#[derive(Debug)]
pub enum Expr {
    ConstInt(i32),
    Unary(UnaryOp, Box<Expr>),
}
impl Node for Expr {
    #[allow(clippy::cast_sign_loss)]
    // not sure if this is even correct.
    fn to_asm(&self) -> asm::Operand {
        match self {
            Expr::ConstInt(i) => asm::Operand::Imm(*i as u32),
            Expr::Unary(..) => todo!(),
        }
    }
}

#[derive(Debug)]
pub enum UnaryOp {
    Complement,
    Negate,
}
impl Node for UnaryOp {
    fn to_asm(&self) -> impl asm::Asm {
        if true { todo!() } else { asm::Instr::Ret }
    }
}

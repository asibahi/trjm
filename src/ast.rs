#![allow(refining_impl_trait_internal)]

use crate::tac;
use ecow::{EcoString, eco_format};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

pub trait Node {
    type Output: tac::Tac;
    fn to_tac(&self, instrs: &mut Vec<tac::Instr>) -> Self::Output;
}

#[derive(Debug)]
pub struct Program(pub FuncDef);
impl Program {
    pub fn compile(&self) -> tac::Program {
        let mut buf = vec![];
        self.to_tac(&mut buf)
    }
}
impl Node for Program {
    type Output = tac::Program;
    fn to_tac(&self, instrs: &mut Vec<tac::Instr>) -> Self::Output {
        tac::Program(self.0.to_tac(instrs))
    }
}

#[derive(Debug)]
pub struct FuncDef {
    pub name: EcoString,
    pub body: Stmt,
}
impl Node for FuncDef {
    type Output = tac::FuncDef;
    fn to_tac(&self, instrs: &mut Vec<tac::Instr>) -> Self::Output {
        Self::Output {
            name: self.name.clone(),
            body: self.body.to_tac(instrs),
        }
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
    type Output = Vec<tac::Instr>;
    fn to_tac(&self, instrs: &mut Vec<tac::Instr>) -> Self::Output {
        match self {
            Stmt::Return(expr) => {
                let dst = expr.to_tac(instrs);

                // is this right?
                let mut ret = std::mem::take(instrs);
                ret.push(tac::Instr::Return(dst));

                ret
            }
            Stmt::If { cond, then, else_ } => unimplemented!(),
        }
    }
}

#[derive(Debug)]
pub enum Expr {
    ConstInt(i32),
    Unary(UnaryOp, Box<Expr>),
}
impl Node for Expr {
    type Output = tac::Value;
    #[allow(clippy::cast_sign_loss)]
    fn to_tac(&self, instrs: &mut Vec<tac::Instr>) -> Self::Output {
        match self {
            Expr::ConstInt(i) => tac::Value::Const(*i as u32),
            Expr::Unary(unary_op, expr) => {
                static UNARY_TMP: AtomicUsize = AtomicUsize::new(0);

                let src = expr.to_tac(instrs);

                let dst_name = eco_format!("unop.tmp.{}", UNARY_TMP.fetch_add(1, Relaxed));
                let dst = tac::Place(dst_name);
                let op = unary_op.to_tac(instrs);

                instrs.push(tac::Instr::Unary {
                    op,
                    src,
                    dst: dst.clone(),
                });

                tac::Value::Var(dst)
            }
        }
    }
}

#[derive(Debug)]
pub enum UnaryOp {
    Complement,
    Negate,
}
impl Node for UnaryOp {
    type Output = tac::UnOp;
    fn to_tac(&self, _: &mut Vec<tac::Instr>) -> Self::Output {
        match self {
            UnaryOp::Complement => tac::UnOp::Complement,
            UnaryOp::Negate => tac::UnOp::Negate,
        }
    }
}

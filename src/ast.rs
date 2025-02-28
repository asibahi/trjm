use crate::tac;
use ecow::{EcoString, eco_format};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

pub trait Node {
    type Output: tac::Tac;
    fn to_tac(&self, instrs: &mut Vec<tac::Instr>) -> Self::Output;
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub enum Stmt {
    Return(Box<Expr>),
    #[expect(unused)]
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
            Stmt::If { .. } => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    ConstInt(i32),
    Unary(UnaryOp, Box<Expr>),
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    #[allow(unused)]
    Conditional {
        cond: Box<Expr>,
        then: Box<Expr>,
        else_: Box<Expr>,
    },
}
impl Node for Expr {
    type Output = tac::Value;
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::too_many_lines)]
    fn to_tac(&self, instrs: &mut Vec<tac::Instr>) -> Self::Output {
        match self {
            Expr::ConstInt(i) => tac::Value::Const(*i as u32),
            Expr::Unary(UnaryOp::Plus, expr) => expr.to_tac(instrs),
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
            Expr::Binary {
                op: BinaryOp::And,
                lhs,
                rhs,
            } => {
                static AND: AtomicUsize = AtomicUsize::new(0);
                let counter = AND.fetch_add(1, Relaxed);

                let if_false = eco_format!("and.fls.{}", counter);
                let end = eco_format!("and.end.{}", counter);

                let dst_name = eco_format!("and.tmp.{}", counter);
                let dst = tac::Place(dst_name);

                let v1 = lhs.to_tac(instrs);
                instrs.push(tac::Instr::JumpIfZero {
                    cond: v1,
                    target: if_false.clone(),
                });

                let v2 = rhs.to_tac(instrs);
                instrs.extend([
                    tac::Instr::JumpIfZero {
                        cond: v2,
                        target: if_false.clone(),
                    },
                    tac::Instr::Copy {
                        src: tac::Value::Const(1),
                        dst: dst.clone(),
                    },
                    tac::Instr::Jump {
                        target: end.clone(),
                    },
                    tac::Instr::Label(if_false),
                    tac::Instr::Copy {
                        src: tac::Value::Const(0),
                        dst: dst.clone(),
                    },
                    tac::Instr::Label(end),
                ]);

                tac::Value::Var(dst)
            }
            Expr::Binary {
                op: BinaryOp::Or,
                lhs,
                rhs,
            } => {
                static OR: AtomicUsize = AtomicUsize::new(0);
                let counter = OR.fetch_add(1, Relaxed);

                let if_true = eco_format!("or.tru.{}", counter);
                let end = eco_format!("or.end.{}", counter);

                let dst_name = eco_format!("or.tmp.{}", counter);
                let dst = tac::Place(dst_name);

                let v1 = lhs.to_tac(instrs);
                instrs.push(tac::Instr::JumpIfNotZero {
                    cond: v1,
                    target: if_true.clone(),
                });

                let v2 = rhs.to_tac(instrs);
                instrs.extend([
                    tac::Instr::JumpIfNotZero {
                        cond: v2,
                        target: if_true.clone(),
                    },
                    tac::Instr::Copy {
                        src: tac::Value::Const(0),
                        dst: dst.clone(),
                    },
                    tac::Instr::Jump {
                        target: end.clone(),
                    },
                    tac::Instr::Label(if_true),
                    tac::Instr::Copy {
                        src: tac::Value::Const(1),
                        dst: dst.clone(),
                    },
                    tac::Instr::Label(end),
                ]);

                tac::Value::Var(dst)
            }
            Expr::Binary { op, lhs, rhs } => {
                static BINARY_TMP: AtomicUsize = AtomicUsize::new(0);

                let src1 = lhs.to_tac(instrs);
                let src2 = rhs.to_tac(instrs);

                let dst_name = eco_format!("binop.tmp.{}", BINARY_TMP.fetch_add(1, Relaxed));
                let dst = tac::Place(dst_name);

                let op = op.to_tac(instrs);

                instrs.push(tac::Instr::Binary {
                    op,
                    src1,
                    src2,
                    dst: dst.clone(),
                });
                tac::Value::Var(dst)
            }
            Expr::Conditional { .. } => todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Complement,
    Negate,
    Not,

    // extra credit
    Plus,
}
impl Node for UnaryOp {
    type Output = tac::UnOp;
    fn to_tac(&self, _: &mut Vec<tac::Instr>) -> Self::Output {
        match self {
            UnaryOp::Complement => tac::UnOp::Complement,
            UnaryOp::Negate => tac::UnOp::Negate,
            UnaryOp::Not => tac::UnOp::Not,

            UnaryOp::Plus => unreachable!("noop operation"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Reminder,

    And,
    Or,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,

    // extra credit
    BitAnd,
    BitOr,
    BitXor,
    LeftShift,
    RightShift,
}
impl Node for BinaryOp {
    type Output = tac::BinOp;
    fn to_tac(&self, _: &mut Vec<tac::Instr>) -> Self::Output {
        match self {
            BinaryOp::Add => tac::BinOp::Add,
            BinaryOp::Subtract => tac::BinOp::Subtract,
            BinaryOp::Multiply => tac::BinOp::Multiply,
            BinaryOp::Divide => tac::BinOp::Divide,
            BinaryOp::Reminder => tac::BinOp::Reminder,

            // chapter 4
            BinaryOp::And | BinaryOp::Or => unreachable!("And and Or have special logic"),
            BinaryOp::Equal => tac::BinOp::Equal,
            BinaryOp::NotEqual => tac::BinOp::NotEqual,
            BinaryOp::LessThan => tac::BinOp::LessThan,
            BinaryOp::LessOrEqual => tac::BinOp::LessOrEqual,
            BinaryOp::GreaterThan => tac::BinOp::GreaterThan,
            BinaryOp::GreaterOrEqual => tac::BinOp::GreaterOrEqual,

            // extra credit
            BinaryOp::BitAnd => tac::BinOp::BitAnd,
            BinaryOp::BitOr => tac::BinOp::BitOr,
            BinaryOp::BitXor => tac::BinOp::BitXor,
            BinaryOp::LeftShift => tac::BinOp::LeftShift,
            BinaryOp::RightShift => tac::BinOp::RightShift,
        }
    }
}

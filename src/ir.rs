use crate::{assembly, ast};
use ecow::{EcoString, eco_format};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

#[derive(Debug, Clone)]
pub struct Program(pub FuncDef);

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: EcoString,
    pub body: Vec<Instr>,
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
        lhs: Value,
        rhs: Value,
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

#[derive(Debug, Clone)]
pub enum Value {
    Const(u32),
    Var(Place),
}

#[derive(Debug, Clone)]
pub struct Place(pub EcoString);

#[derive(Debug, Clone, Copy)]
pub enum UnOp {
    Complement,
    Negate,
    Not,
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

// ======

pub trait ToIr {
    type Output: assembly::ToAsm;
    fn to_ir(&self, instrs: &mut Vec<Instr>) -> Self::Output;
}
impl ToIr for () {
    type Output = ();
    fn to_ir(&self, _: &mut Vec<Instr>) -> Self::Output {}
}
impl ToIr for ast::Program {
    type Output = Program;
    fn to_ir(&self, instrs: &mut Vec<Instr>) -> Self::Output {
        Program(self.0.to_ir(instrs))
    }
}
impl ToIr for ast::FuncDef {
    type Output = FuncDef;
    fn to_ir(&self, instrs: &mut Vec<Instr>) -> Self::Output {
        for bi in &self.body {
            bi.to_ir(instrs);
        }

        if !matches!(instrs.last(), Some(Instr::Return(_))) {
            instrs.push(Instr::Return(Value::Const(0)));
        }

        Self::Output {
            name: self.name.clone(),
            body: std::mem::take(instrs),
        }
    }
}
impl ToIr for ast::Stmt {
    type Output = ();
    fn to_ir(&self, instrs: &mut Vec<Instr>) -> Self::Output {
        match self {
            Self::Return(expr) => {
                let dst = expr.to_ir(instrs);
                instrs.push(Instr::Return(dst));
            }
            Self::Expression(expr) => {
                expr.to_ir(instrs);
            }
            Self::Null => {}
            Self::If { .. } => unimplemented!(),
        }
    }
}
impl ToIr for ast::Expr {
    type Output = Value;
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::too_many_lines)]
    fn to_ir(&self, instrs: &mut Vec<Instr>) -> Self::Output {
        match self {
            Self::ConstInt(i) => Value::Const(*i as u32),
            Self::Unary(ast::UnaryOp::Plus, expr) => expr.to_ir(instrs),
            Self::Unary(
                op @ (ast::UnaryOp::IncPost
                | ast::UnaryOp::DecPost
                | ast::UnaryOp::IncPre
                | ast::UnaryOp::DecPre),
                expr,
            ) => {
                static FIX: AtomicUsize = AtomicUsize::new(0);

                let ast::Expr::Var(ref dst) = **expr else {
                    unreachable!("place expression should be resolved earlier.")
                };

                let (op, cache) = match op {
                    ast::UnaryOp::IncPost => (ast::BinaryOp::Add, true),
                    ast::UnaryOp::DecPost => (ast::BinaryOp::Subtract, true),

                    ast::UnaryOp::IncPre => (ast::BinaryOp::Add, false),
                    ast::UnaryOp::DecPre => (ast::BinaryOp::Subtract, false),
                    _ => unreachable!(),
                };

                let tmp = eco_format!("fix.tmp.{}", FIX.fetch_add(1, Relaxed));
                let tmp = Place(tmp);

                let var = Place(dst.clone());

                if cache {
                    instrs.push(Instr::Copy {
                        src: Value::Var(var.clone()),
                        dst: tmp.clone(),
                    });
                }

                let op = op.to_ir(instrs);

                instrs.push(Instr::Binary {
                    op,
                    lhs: Value::Var(var.clone()),
                    rhs: Value::Const(1),
                    dst: var.clone(),
                });

                if cache {
                    Value::Var(tmp)
                } else {
                    Value::Var(var)
                }
            }
            Self::Unary(unary_op, expr) => {
                static UNARY_TMP: AtomicUsize = AtomicUsize::new(0);

                let src = expr.to_ir(instrs);

                let dst_name = eco_format!("unop.tmp.{}", UNARY_TMP.fetch_add(1, Relaxed));
                let dst = Place(dst_name);
                let op = unary_op.to_ir(instrs);

                instrs.push(Instr::Unary {
                    op,
                    src,
                    dst: dst.clone(),
                });

                Value::Var(dst)
            }
            Self::Binary {
                op: ast::BinaryOp::And,
                lhs,
                rhs,
            } => {
                static AND: AtomicUsize = AtomicUsize::new(0);
                let counter = AND.fetch_add(1, Relaxed);

                let if_false = eco_format!("and.fls.{}", counter);
                let end = eco_format!("and.end.{}", counter);

                let dst_name = eco_format!("and.tmp.{}", counter);
                let dst = Place(dst_name);

                let v1 = lhs.to_ir(instrs);
                instrs.push(Instr::JumpIfZero {
                    cond: v1,
                    target: if_false.clone(),
                });

                let v2 = rhs.to_ir(instrs);
                instrs.extend([
                    Instr::JumpIfZero {
                        cond: v2,
                        target: if_false.clone(),
                    },
                    Instr::Copy {
                        src: Value::Const(1),
                        dst: dst.clone(),
                    },
                    Instr::Jump {
                        target: end.clone(),
                    },
                    Instr::Label(if_false),
                    Instr::Copy {
                        src: Value::Const(0),
                        dst: dst.clone(),
                    },
                    Instr::Label(end),
                ]);

                Value::Var(dst)
            }
            Self::Binary {
                op: ast::BinaryOp::Or,
                lhs,
                rhs,
            } => {
                static OR: AtomicUsize = AtomicUsize::new(0);
                let counter = OR.fetch_add(1, Relaxed);

                let if_true = eco_format!("or.tru.{}", counter);
                let end = eco_format!("or.end.{}", counter);

                let dst_name = eco_format!("or.tmp.{}", counter);
                let dst = Place(dst_name);

                let v1 = lhs.to_ir(instrs);
                instrs.push(Instr::JumpIfNotZero {
                    cond: v1,
                    target: if_true.clone(),
                });

                let v2 = rhs.to_ir(instrs);
                instrs.extend([
                    Instr::JumpIfNotZero {
                        cond: v2,
                        target: if_true.clone(),
                    },
                    Instr::Copy {
                        src: Value::Const(0),
                        dst: dst.clone(),
                    },
                    Instr::Jump {
                        target: end.clone(),
                    },
                    Instr::Label(if_true),
                    Instr::Copy {
                        src: Value::Const(1),
                        dst: dst.clone(),
                    },
                    Instr::Label(end),
                ]);

                Value::Var(dst)
            }
            Self::Binary { op, lhs, rhs } => {
                static BINARY_TMP: AtomicUsize = AtomicUsize::new(0);

                let lhs = lhs.to_ir(instrs);
                let rhs = rhs.to_ir(instrs);

                let dst_name = eco_format!("binop.tmp.{}", BINARY_TMP.fetch_add(1, Relaxed));
                let dst = Place(dst_name);

                let op = op.to_ir(instrs);

                instrs.push(Instr::Binary {
                    op,
                    lhs,
                    rhs,
                    dst: dst.clone(),
                });
                Value::Var(dst)
            }

            Self::Var(id) => Value::Var(Place(id.clone())),
            Self::Assignemnt(place, value) => {
                let ast::Expr::Var(ref dst) = **place else {
                    unreachable!("place expression should be resolved earlier.")
                };

                let rhs = value.to_ir(instrs);

                instrs.push(Instr::Copy {
                    src: rhs,
                    dst: Place(dst.clone()),
                });

                Value::Var(Place(dst.clone()))
            }
            Self::CompoundAssignment { op, lhs, rhs } => {
                let ast::Expr::Var(ref dst) = **lhs else {
                    unreachable!("place expression should be resolved earlier.")
                };
                let ret = Self::Binary {
                    op: *op,
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                }
                .to_ir(instrs);

                instrs.push(Instr::Copy {
                    src: ret,
                    dst: Place(dst.clone()),
                });

                Value::Var(Place(dst.clone()))
            }
            Self::Conditional { .. } => unimplemented!(),
        }
    }
}
impl ToIr for ast::UnaryOp {
    type Output = UnOp;
    fn to_ir(&self, _: &mut Vec<Instr>) -> Self::Output {
        match self {
            Self::Complement => UnOp::Complement,
            Self::Negate => UnOp::Negate,
            Self::Not => UnOp::Not,

            Self::Plus | Self::IncPre | Self::IncPost | Self::DecPre | Self::DecPost => {
                unreachable!("implemented in expr.to_ir")
            }
        }
    }
}
impl ToIr for ast::BinaryOp {
    type Output = BinOp;
    fn to_ir(&self, _: &mut Vec<Instr>) -> Self::Output {
        match self {
            Self::Add => BinOp::Add,
            Self::Subtract => BinOp::Subtract,
            Self::Multiply => BinOp::Multiply,
            Self::Divide => BinOp::Divide,
            Self::Reminder => BinOp::Reminder,

            // chapter 4
            Self::And | Self::Or => unreachable!("And and Or have special logic"),
            Self::Equal => BinOp::Equal,
            Self::NotEqual => BinOp::NotEqual,
            Self::LessThan => BinOp::LessThan,
            Self::LessOrEqual => BinOp::LessOrEqual,
            Self::GreaterThan => BinOp::GreaterThan,
            Self::GreaterOrEqual => BinOp::GreaterOrEqual,

            // extra credit
            Self::BitAnd => BinOp::BitAnd,
            Self::BitOr => BinOp::BitOr,
            Self::BitXor => BinOp::BitXor,
            Self::LeftShift => BinOp::LeftShift,
            Self::RightShift => BinOp::RightShift,
        }
    }
}
impl ToIr for ast::BlockItem {
    type Output = ();

    fn to_ir(&self, instrs: &mut Vec<Instr>) -> Self::Output {
        match self {
            ast::BlockItem::S(stmt) => {
                stmt.to_ir(instrs);
            }
            ast::BlockItem::D(decl) => {
                decl.to_ir(instrs);
            }
        }
    }
}
impl ToIr for ast::Decl {
    type Output = ();

    fn to_ir(&self, instrs: &mut Vec<Instr>) -> Self::Output {
        if let Some(e) = &self.init {
            let v = e.to_ir(instrs);
            instrs.push(Instr::Copy {
                src: v,
                dst: Place(self.name.clone()),
            });
        }
    }
}

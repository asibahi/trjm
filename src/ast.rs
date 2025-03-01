#![allow(clippy::zero_sized_map_values)]

use crate::ir::{self, ToIr};
use ecow::{EcoString, eco_format};
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

#[derive(Debug, Clone)]
pub struct Program(pub FuncDef);
impl Program {
    pub fn compile(&self) -> ir::Program {
        let mut buf = vec![];
        self.to_ir(&mut buf)
    }
    pub fn resolve_namespaces(self) -> Option<Self> {
        let mut var_map = FxHashMap::default();

        let s1 = Self(self.0.resolve_variables(&mut var_map)?);
        let s2 = Self(s1.0.resolve_labels()?);

        Some(s2)
    }
}

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: EcoString,
    pub body: Vec<BlockItem>,
}
impl FuncDef {
    fn resolve_variables(self, map: &mut Namespace<EcoString>) -> Option<Self> {
        let mut acc = Vec::with_capacity(self.body.len());

        for bi in self.body {
            let bi = bi.resolve_variables(map)?;
            acc.push(bi);
        }

        Some(Self { name: self.name, body: acc })
    }

    fn resolve_labels(self) -> Option<Self> {
        let mut label_map = FxHashMap::default();
        let mut acc = Vec::with_capacity(self.body.len());

        for bi in self.body {
            let bi = bi.resolve_labels(&mut label_map)?;
            acc.push(bi);
        }

        if label_map.values().any(|v| !(*v)) {
            return None;
        };

        Some(Self { name: self.name, body: acc })
    }
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Return(Expr),
    Expression(Expr),
    If { cond: Expr, then: Box<Stmt>, else_: Option<Box<Stmt>> },
    GoTo(EcoString),
    Label(EcoString, Box<Stmt>),

    Null,
}
type Namespace<T> = FxHashMap<EcoString, T>;
impl Stmt {
    fn resolve_variables(self, map: &mut Namespace<EcoString>) -> Option<Self> {
        match self {
            Self::Return(expr) => Some(Self::Return(expr.resolve_variables(map)?)),
            Self::Expression(expr) => Some(Self::Expression(expr.resolve_variables(map)?)),
            Self::If { cond, then, else_ } => {
                let else_ = match else_ {
                    Some(s) => Some(Box::new(s.resolve_variables(map)?)),
                    None => None,
                };
                Some(Self::If {
                    cond: cond.resolve_variables(map)?,
                    then: Box::new(then.resolve_variables(map)?),
                    else_,
                })
            }

            Self::Label(name, stmt) => {
                Some(Self::Label(name, Box::new(stmt.resolve_variables(map)?)))
            }
            g @ Self::GoTo(_) => Some(g),
            n @ Self::Null => Some(n),
        }
    }
    fn resolve_labels(self, labels: &mut Namespace<bool>) -> Option<Self> {
        match self {
            Self::GoTo(label) => {
                if !labels.contains_key(&label) {
                    labels.insert(label.clone(), false);
                }

                Some(Self::GoTo(label))
            }
            Self::Label(label, s) => {
                if labels.get(&label).is_some_and(|v| *v) {
                    eprintln!("repeated label : {label}");

                    return None;
                }
                labels.insert(label.clone(), true);
                let s = s.resolve_labels(labels)?;

                Some(Self::Label(label, Box::new(s)))
            }
            Self::If { cond, then, else_ } => {
                let then = Box::new(then.resolve_labels(labels)?);
                let else_ = match else_ {
                    Some(s) => Some(Box::new(s.resolve_labels(labels)?)),
                    None => None,
                };

                Some(Self::If { cond, then, else_ })
            }
            any => Some(any),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    ConstInt(i32),
    Var(EcoString),
    Unary(UnaryOp, Box<Expr>),
    Binary { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    CompoundAssignment { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Assignemnt(Box<Expr>, Box<Expr>),
    Conditional { cond: Box<Expr>, then: Box<Expr>, else_: Box<Expr> },
}
impl Expr {
    fn resolve_variables(self, map: &mut Namespace<EcoString>) -> Option<Self> {
        match self {
            Self::Assignemnt(left, right) if matches!(*left, Expr::Var(_)) => {
                Some(Self::Assignemnt(
                    Box::new(left.resolve_variables(map)?),
                    Box::new(right.resolve_variables(map)?),
                ))
            }
            Self::Assignemnt(_, _) => None,

            // magic happens here
            Self::Var(var) => map.get(&var).cloned().map(Self::Var),

            Self::Unary(op, expr) => Some(Self::Unary(op, Box::new(expr.resolve_variables(map)?))),

            Self::Binary { op, lhs, rhs } => Some(Self::Binary {
                op,
                lhs: Box::new(lhs.resolve_variables(map)?),
                rhs: Box::new(rhs.resolve_variables(map)?),
            }),

            Self::CompoundAssignment { op, lhs, rhs } => Some(Self::CompoundAssignment {
                op,
                lhs: Box::new(lhs.resolve_variables(map)?),
                rhs: Box::new(rhs.resolve_variables(map)?),
            }),

            v @ Self::ConstInt(_) => Some(v),
            Self::Conditional { cond, then, else_ } => Some(Self::Conditional {
                cond: Box::new(cond.resolve_variables(map)?),
                then: Box::new(then.resolve_variables(map)?),
                else_: Box::new(else_.resolve_variables(map)?),
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Decl {
    pub name: EcoString,
    pub init: Option<Expr>,
}
impl Decl {
    fn resolve_variables(self, map: &mut Namespace<EcoString>) -> Option<Self> {
        static DECL: AtomicUsize = AtomicUsize::new(0);

        if map.contains_key(&self.name) {
            return None;
        }

        let unique_name = eco_format!("{}.{}", self.name, DECL.fetch_add(1, Relaxed));
        map.insert(self.name, unique_name.clone());

        let init =
            if let Some(init) = self.init { Some(init.resolve_variables(map)?) } else { None };

        Some(Decl { name: unique_name, init })
    }
}

#[derive(Debug, Clone)]
pub enum BlockItem {
    S(Stmt),
    D(Decl),
}
impl BlockItem {
    fn resolve_variables(self, map: &mut Namespace<EcoString>) -> Option<Self> {
        match self {
            Self::S(stmt) => Some(Self::S(stmt.resolve_variables(map)?)),
            Self::D(decl) => Some(Self::D(decl.resolve_variables(map)?)),
        }
    }
    fn resolve_labels(self, labels: &mut Namespace<bool>) -> Option<Self> {
        match self {
            Self::S(stmt) => Some(Self::S(stmt.resolve_labels(labels)?)),
            d @ Self::D(_) => Some(d),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Complement,
    Negate,
    Not,

    // pedantic
    Plus,

    // chapter 5 extra credit
    IncPre,
    IncPost,
    DecPre,
    DecPost,
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

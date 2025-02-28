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
    pub fn resolve_variables(
        self,
    ) -> Option<Self> {
        let mut variable_map = FxHashMap::default();
        Some(Self(self.0.resolve_variables(&mut variable_map)?))
    }
}

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: EcoString,
    pub body: Vec<BlockItem>,
}
impl FuncDef {
    fn resolve_variables(self, variable_map: &mut FxHashMap<EcoString, EcoString>) -> Option<Self> {
        let mut acc = Vec::with_capacity(self.body.len());

        for bi in self.body {
            let bi = bi.resolve_variables(variable_map)?;
            acc.push(bi);
        }

        Some(Self {
            name: self.name,
            body: acc,
        })
    }
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Return(Expr),
    Expression(Expr),
    Null,
    #[expect(unused)]
    If {
        cond: Expr,
        then: Box<Stmt>,
        else_: Option<Box<Stmt>>,
    },
}
impl Stmt {
    fn resolve_variables(self, variable_map: &mut FxHashMap<EcoString, EcoString>) -> Option<Self> {
        match self {
            Self::Return(expr) => Some(Self::Return(expr.resolve_variables(variable_map)?)),
            Self::Expression(expr) => Some(Self::Expression(expr.resolve_variables(variable_map)?)),
            Self::Null => Some(Self::Null),
            Self::If { .. } => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    ConstInt(i32),
    Var(EcoString),
    Unary(UnaryOp, Box<Expr>),
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Assignemnt(Box<Expr>, Box<Expr>),
    #[allow(unused)]
    Conditional {
        cond: Box<Expr>,
        then: Box<Expr>,
        else_: Box<Expr>,
    },
}
impl Expr {
    fn resolve_variables(self, variable_map: &mut FxHashMap<EcoString, EcoString>) -> Option<Self> {
        match self {
            Self::Assignemnt(left, right) if matches!(*left, Expr::Var(_)) => {
                Some(Self::Assignemnt(
                    Box::new(left.resolve_variables(variable_map)?),
                    Box::new(right.resolve_variables(variable_map)?),
                ))
            }
            Self::Assignemnt(_, _) => None,

            Self::Var(var) => variable_map.get(&var).cloned().map(Self::Var),

            Self::Unary(op, expr) => Some(Self::Unary(
                op,
                Box::new(expr.resolve_variables(variable_map)?),
            )),

            Self::Binary { op, lhs, rhs } => Some(Self::Binary {
                op,
                lhs: Box::new(lhs.resolve_variables(variable_map)?),
                rhs: Box::new(rhs.resolve_variables(variable_map)?),
            }),

            v @ Self::ConstInt(_) => Some(v),
            Self::Conditional { .. } => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Decl {
    pub name: EcoString,
    pub init: Option<Expr>,
}
impl Decl {
    fn resolve_variables(self, variable_map: &mut FxHashMap<EcoString, EcoString>) -> Option<Self> {
        static DECL: AtomicUsize = AtomicUsize::new(0);

        if variable_map.contains_key(&self.name) {
            return None;
        }

        let unique_name = eco_format!("{}.{}", self.name, DECL.fetch_add(1, Relaxed));
        variable_map.insert(self.name, unique_name.clone());

        let init = if let Some(init) = self.init {
            Some(init.resolve_variables(variable_map)?)
        } else {
            None
        };

        Some(Decl {
            name: unique_name,
            init,
        })
    }
}

#[derive(Debug, Clone)]
pub enum BlockItem {
    S(Stmt),
    D(Decl),
}
impl BlockItem {
    fn resolve_variables(self, variable_map: &mut FxHashMap<EcoString, EcoString>) -> Option<Self> {
        match self {
            Self::S(stmt) => Some(Self::S(stmt.resolve_variables(variable_map)?)),
            Self::D(decl) => Some(Self::D(decl.resolve_variables(variable_map)?)),
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

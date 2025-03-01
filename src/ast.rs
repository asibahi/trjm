#![allow(clippy::zero_sized_map_values)]

use crate::ir::{self, ToIr};
use ecow::{EcoString, eco_format};
use either::Either::{self, Left, Right};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    collections::HashSet,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

type Namespace<T> = FxHashMap<EcoString, T>;

#[derive(Debug, Clone)]
pub struct Program(pub FuncDef);
impl Program {
    pub fn compile(&self) -> ir::Program {
        let mut buf = vec![];
        self.to_ir(&mut buf)
    }
    pub fn resolve_resolutions(self) -> Option<Self> {
        let mut var_map = FxHashMap::default();

        let step_0 = Self(self.0.resolve_switch_statements()?);

        let step_1 = Self(step_0.0.resolve_variables(&mut var_map)?);
        let step_2 = Self(step_1.0.resolve_goto_labels()?);

        let step_3 = Self(step_2.0.resolve_loop_labels()?);

        Some(step_3)
    }
}

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: EcoString,
    pub body: Block,
}
impl FuncDef {
    fn resolve_variables(self, map: &mut Namespace<(EcoString, bool)>) -> Option<Self> {
        let body = self.body.resolve_variables(map)?;

        Some(Self { name: self.name, body })
    }

    fn resolve_goto_labels(self) -> Option<Self> {
        // labels are function level
        let mut label_map = FxHashMap::default();
        let body = self.body.resolve_goto_labels(&mut label_map)?;

        if label_map.values().any(|v| !(*v)) {
            return None;
        };

        Some(Self { name: self.name, body })
    }

    fn resolve_loop_labels(self) -> Option<Self> {
        let body = self.body.resolve_loop_labels(LK::None)?;

        Some(Self { name: self.name, body })
    }

    fn resolve_switch_statements(self) -> Option<Self> {
        let body = self.body.resolve_switch_statements(None)?;

        Some(Self { name: self.name, body })
    }
}

#[derive(Debug, Clone)]
enum LK {
    Loop(EcoString),
    Switch(EcoString),
    SwitchInLoop { loop_: EcoString, switch: EcoString },
    LoopInSwitch { loop_: EcoString, switch: EcoString },
    None,
}
impl LK {
    fn break_label(&self) -> Option<EcoString> {
        match self {
            LK::Loop(n)
            | LK::Switch(n)
            | LK::SwitchInLoop { switch: n, .. }
            | LK::LoopInSwitch { loop_: n, .. } => Some(n.clone()),
            LK::None => None,
        }
    }
    fn loop_label(&self) -> Option<EcoString> {
        match self {
            LK::Loop(n) | LK::SwitchInLoop { loop_: n, .. } | LK::LoopInSwitch { loop_: n, .. } => {
                Some(n.clone())
            }
            LK::None | LK::Switch(_) => None,
        }
    }
    fn switch_label(&self) -> Option<EcoString> {
        match self {
            LK::Switch(n)
            | LK::SwitchInLoop { switch: n, .. }
            | LK::LoopInSwitch { switch: n, .. } => Some(n.clone()),
            LK::None | LK::Loop(_) => None,
        }
    }
    fn into_switch(self, label: EcoString) -> Self {
        match self {
            LK::SwitchInLoop { loop_, .. } | LK::LoopInSwitch { loop_, .. } | LK::Loop(loop_) => {
                LK::SwitchInLoop { loop_, switch: label }
            }
            LK::Switch(_) | LK::None => LK::Switch(label),
        }
    }
    fn into_loop(self, label: EcoString) -> Self {
        match self {
            LK::SwitchInLoop { switch, .. }
            | LK::Switch(switch)
            | LK::LoopInSwitch { switch, .. } => LK::LoopInSwitch { loop_: label, switch },
            LK::None | LK::Loop(_) => LK::Loop(label),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Return(Expr),
    Expression(Expr),
    If {
        cond: Expr,
        then: Box<Stmt>,
        else_: Option<Box<Stmt>>,
    },

    Compound(Block),

    Break(Option<EcoString>),
    Continue(Option<EcoString>),
    While {
        cond: Expr,
        body: Box<Stmt>,
        label: Option<EcoString>,
    },
    DoWhile {
        body: Box<Stmt>,
        cond: Expr,
        label: Option<EcoString>,
    },
    For {
        init: Either<Decl, Option<Expr>>,
        cond: Option<Expr>,
        post: Option<Expr>,

        body: Box<Stmt>,
        label: Option<EcoString>,
    },

    // extra credit
    GoTo(EcoString),
    Label(EcoString, Box<Stmt>),

    Switch {
        ctrl: Expr,
        body: Box<Stmt>,
        label: Option<EcoString>,
        cases: Vec<Option<i32>>,
    },
    Case {
        cnst: Expr,
        body: Box<Stmt>,
        label: Option<EcoString>,
    },
    Default {
        body: Box<Stmt>,
        label: Option<EcoString>,
    },

    Null,
}
impl Stmt {
    fn resolve_variables(self, map: &mut Namespace<(EcoString, bool)>) -> Option<Self> {
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

            Self::Compound(block) => Some(Self::Compound(block.resolve_variables(map)?)),
            Self::While { cond, body, label } => {
                let cond = cond.resolve_variables(map)?;
                let body = Box::new(body.resolve_variables(map)?);

                Some(Self::While { cond, body, label: label.clone() })
            }
            Self::DoWhile { cond, body, label } => {
                let cond = cond.resolve_variables(map)?;
                let body = Box::new(body.resolve_variables(map)?);

                Some(Self::DoWhile { cond, body, label: label.clone() })
            }
            Self::For { init, cond, post, body, label } => {
                let mut for_map =
                    map.iter().map(|(k, (v, _))| (k.clone(), (v.clone(), false))).collect();

                let init = match init {
                    Left(decl) => Left(decl.resolve_variables(&mut for_map)?),
                    Right(Some(expr)) => Right(Some(expr.resolve_variables(&mut for_map)?)),
                    Right(None) => Right(None),
                };
                let cond = match cond {
                    Some(cond) => Some(cond.resolve_variables(&mut for_map)?),
                    None => None,
                };
                let post = match post {
                    Some(post) => Some(post.resolve_variables(&mut for_map)?),
                    None => None,
                };

                let body = Box::new(body.resolve_variables(&mut for_map)?);

                Some(Self::For { init, cond, post, body, label: label.clone() })
            }

            Self::Switch { ctrl, body, label, cases } => {
                let ctrl = ctrl.resolve_variables(map)?;
                let body = Box::new(body.resolve_variables(map)?);

                Some(Self::Switch { ctrl, body, label, cases })
            }
            Self::Case { cnst, body, label } => {
                let body = Box::new(body.resolve_variables(map)?);

                Some(Self::Case { cnst, body, label })
            }
            Self::Default { body, label } => {
                Some(Self::Default { body: Box::new(body.resolve_variables(map)?), label })
            }

            n @ (Self::Null | Self::Break(_) | Self::Continue(_)) => Some(n),
        }
    }
    fn resolve_goto_labels(self, labels: &mut Namespace<bool>) -> Option<Self> {
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
                let s = s.resolve_goto_labels(labels)?;

                Some(Self::Label(label, Box::new(s)))
            }
            Self::If { cond, then, else_ } => {
                let then = Box::new(then.resolve_goto_labels(labels)?);
                let else_ = match else_ {
                    Some(s) => Some(Box::new(s.resolve_goto_labels(labels)?)),
                    None => None,
                };

                Some(Self::If { cond, then, else_ })
            }
            Self::Compound(block) => Some(Self::Compound(block.resolve_goto_labels(labels)?)),
            any @ (Self::Return(_)
            | Self::Null
            | Self::Expression(_)
            | Self::Break(_)
            | Self::Continue(_)) => Some(any),

            Self::While { body, cond, label } => {
                Some(Self::While { body: Box::new(body.resolve_goto_labels(labels)?), cond, label })
            }
            Self::DoWhile { body, cond, label } => Some(Self::DoWhile {
                body: Box::new(body.resolve_goto_labels(labels)?),
                cond,
                label,
            }),
            Self::For { init, cond, post, body, label } => Some(Self::For {
                init,
                cond,
                post,
                body: Box::new(body.resolve_goto_labels(labels)?),
                label,
            }),

            // extra extra credit // Incomplete
            Self::Switch { ctrl, body, label, cases } => Some(Self::Switch {
                ctrl,
                body: Box::new(body.resolve_goto_labels(labels)?),
                label,
                cases,
            }),
            Self::Case { cnst, body, label } => {
                Some(Self::Case { cnst, body: Box::new(body.resolve_goto_labels(labels)?), label })
            }
            Self::Default { body, label } => {
                Some(Self::Default { body: Box::new(body.resolve_goto_labels(labels)?), label })
            }
        }
    }

    fn resolve_loop_labels(self, current_label: LK) -> Option<Stmt> {
        static LABEL: AtomicUsize = AtomicUsize::new(0);
        let loop_counter = LABEL.fetch_add(1, Relaxed);

        match self {
            Self::Break(_) => Some(Self::Break(Some(current_label.break_label()?))),
            Self::Continue(_) => Some(Self::Continue(Some(current_label.loop_label()?))),

            Self::While { cond, body, .. } => {
                let new_label = current_label.into_loop(eco_format!("while.{}", loop_counter));
                let body = Box::new(body.resolve_loop_labels(new_label.clone())?);

                Some(Self::While { cond, body, label: new_label.break_label() })
            }
            Self::DoWhile { cond, body, .. } => {
                let new_label = current_label.into_loop(eco_format!("do.{}", loop_counter));
                let body = Box::new(body.resolve_loop_labels(new_label.clone())?);

                Some(Self::DoWhile { cond, body, label: new_label.break_label() })
            }
            Self::For { init, cond, post, body, .. } => {
                let new_label = current_label.into_loop(eco_format!("for.{}", loop_counter));
                let body = Box::new(body.resolve_loop_labels(new_label.clone())?);

                Some(Self::For { init, cond, post, body, label: new_label.break_label() })
            }
            Self::Switch { ctrl, body, cases, .. } => {
                let new_label = current_label.into_switch(eco_format!("swch.{}", loop_counter));

                let body = Box::new(body.resolve_loop_labels(new_label.clone())?);

                Some(Self::Switch { ctrl, body, label: new_label.break_label(), cases })
            }
            Self::Case { cnst, body, .. } => Some(Self::Case {
                cnst,
                body: Box::new(body.resolve_loop_labels(current_label.clone())?),
                label: current_label.switch_label(),
            }),
            Self::Default { body, .. } => Some(Self::Default {
                body: Box::new(body.resolve_loop_labels(current_label.clone())?),
                label: current_label.switch_label(),
            }),

            Self::If { cond, then, else_ } => {
                let then = Box::new(then.resolve_loop_labels(current_label.clone())?);
                let else_ = match else_ {
                    Some(stmt) => Some(Box::new(stmt.resolve_loop_labels(current_label)?)),
                    None => None,
                };

                Some(Self::If { cond, then, else_ })
            }
            Self::Compound(block) => {
                Some(Self::Compound(block.resolve_loop_labels(current_label)?))
            }
            Self::Label(label, stmt) => {
                let stmt = Box::new(stmt.resolve_loop_labels(current_label)?);
                Some(Self::Label(label, stmt))
            }

            Self::Return(_) | Self::Expression(_) | Self::GoTo(_) | Self::Null => Some(self),
        }
    }

    fn resolve_switch_statements(self, switch_ctx: SwitchCtx) -> Option<Self> {
        Some(match self {
            Self::If { cond, then, else_ } => {
                let mut switch_ctx = switch_ctx;

                let then = Box::new(then.resolve_switch_statements(switch_ctx.as_deref_mut())?);
                let else_ = match else_ {
                    Some(e) => Some(Box::new(e.resolve_switch_statements(switch_ctx)?)),
                    None => None,
                };

                Self::If { cond, then, else_ }
            }
            Self::Compound(block) => Self::Compound(block.resolve_switch_statements(switch_ctx)?),
            Self::While { cond, body, label } => {
                let body = Box::new(body.resolve_switch_statements(switch_ctx)?);
                Self::While { cond, body, label }
            }
            Self::DoWhile { body, cond, label } => {
                let body = Box::new(body.resolve_switch_statements(switch_ctx)?);
                Self::DoWhile { cond, body, label }
            }
            Self::For { init, cond, post, body, label } => {
                let body = Box::new(body.resolve_switch_statements(switch_ctx)?);
                Self::For { init, cond, post, body, label }
            }
            Self::Label(label, stmt) => {
                let stmt = Box::new(stmt.resolve_switch_statements(switch_ctx)?);
                Self::Label(label, stmt)
            }

            Self::Switch { ctrl, body, label, .. } => {
                let mut switch_ctx = HashSet::default();
                let body = Box::new(body.resolve_switch_statements(Some(&mut switch_ctx))?);

                Self::Switch { ctrl, body, label, cases: switch_ctx.into_iter().collect() }
            }
            Self::Case { cnst, body, label } => {
                let switch_ctx = switch_ctx?;

                let Expr::ConstInt(value) = cnst else {
                    return None;
                };

                if !switch_ctx.insert(Some(value)) {
                    return None;
                };

                let body = Box::new(body.resolve_switch_statements(Some(switch_ctx))?);
                Self::Case { cnst, body, label }
            }
            Self::Default { body, label } => {
                let switch_ctx = switch_ctx?;
                if !switch_ctx.insert(None) {
                    return None;
                };

                let body = Box::new(body.resolve_switch_statements(Some(switch_ctx))?);
                Self::Default { body, label }
            }

            Self::Break(_)
            | Self::Continue(_)
            | Self::Return(_)
            | Self::Expression(_)
            | Self::GoTo(_)
            | Self::Null => self,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    fn resolve_variables(self, map: &mut Namespace<(EcoString, bool)>) -> Option<Self> {
        match self {
            Self::Assignemnt(left, right) if matches!(*left, Expr::Var(_)) => {
                Some(Self::Assignemnt(
                    Box::new(left.resolve_variables(map)?),
                    Box::new(right.resolve_variables(map)?),
                ))
            }
            Self::Assignemnt(_, _) => None,

            // magic happens here
            Self::Var(var) => map.get(&var).cloned().map(|t| Self::Var(t.0)),

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
    fn resolve_variables(self, map: &mut Namespace<(EcoString, bool)>) -> Option<Self> {
        static DECL: AtomicUsize = AtomicUsize::new(0);

        if map.get(&self.name).is_some_and(|(_, from_current)| *from_current) {
            return None;
        }

        let unique_name = eco_format!("{}.{}", self.name, DECL.fetch_add(1, Relaxed));

        // shadowing happens here
        map.insert(self.name, (unique_name.clone(), true));

        let init =
            if let Some(init) = self.init { Some(init.resolve_variables(map)?) } else { None };

        Some(Decl { name: unique_name, init })
    }
}

type SwitchCtx<'s> = Option<&'s mut FxHashSet<Option<i32>>>;

#[derive(Debug, Clone)]
pub struct Block(pub Vec<BlockItem>);
impl Block {
    fn resolve_variables(self, map: &mut Namespace<(EcoString, bool)>) -> Option<Self> {
        let mut block_map = map.iter().map(|(k, (v, _))| (k.clone(), (v.clone(), false))).collect();
        let mut acc = Vec::with_capacity(self.0.len());

        for bi in self.0 {
            let bi = bi.resolve_variables(&mut block_map)?;
            acc.push(bi);
        }

        Some(Self(acc))
    }
    fn resolve_goto_labels(self, labels: &mut Namespace<bool>) -> Option<Self> {
        let mut acc = Vec::with_capacity(self.0.len());
        for bi in self.0 {
            let bi = bi.resolve_goto_labels(labels)?;
            acc.push(bi);
        }

        Some(Self(acc))
    }

    fn resolve_loop_labels(self, current_label: LK) -> Option<Self> {
        let mut acc = Vec::with_capacity(self.0.len());
        let current_label = current_label; // pedantic clippy

        for bi in self.0 {
            let bi = bi.resolve_loop_labels(current_label.clone())?;
            acc.push(bi);
        }

        Some(Self(acc))
    }

    fn resolve_switch_statements(self, switch_ctx: SwitchCtx) -> Option<Self> {
        let mut acc = Vec::with_capacity(self.0.len());
        if let Some(switch_ctx) = switch_ctx {
            for bi in self.0 {
                let bi = bi.resolve_switch_statements(Some(switch_ctx))?;
                acc.push(bi);
            }
        } else {
            for bi in self.0 {
                let bi = bi.resolve_switch_statements(None)?;
                acc.push(bi);
            }
        }

        Some(Self(acc))
    }
}
#[derive(Debug, Clone)]
pub enum BlockItem {
    S(Stmt),
    D(Decl),
}
impl BlockItem {
    fn resolve_variables(self, map: &mut Namespace<(EcoString, bool)>) -> Option<Self> {
        match self {
            Self::S(stmt) => Some(Self::S(stmt.resolve_variables(map)?)),
            Self::D(decl) => Some(Self::D(decl.resolve_variables(map)?)),
        }
    }
    fn resolve_goto_labels(self, labels: &mut Namespace<bool>) -> Option<Self> {
        match self {
            Self::S(stmt) => Some(Self::S(stmt.resolve_goto_labels(labels)?)),
            d @ Self::D(_) => Some(d),
        }
    }

    fn resolve_loop_labels(self, current_label: LK) -> Option<Self> {
        Some(match self {
            Self::S(stmt) => Self::S(stmt.resolve_loop_labels(current_label)?),
            Self::D(_) => self,
        })
    }

    fn resolve_switch_statements(self, in_switch: SwitchCtx) -> Option<Self> {
        Some(match self {
            BlockItem::S(stmt) => Self::S(stmt.resolve_switch_statements(in_switch)?),
            BlockItem::D(_) => self,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

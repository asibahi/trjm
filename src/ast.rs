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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct IdCtx {
    name: EcoString,
    in_current_scope: bool,
    has_linkage: bool,
}
impl IdCtx {
    fn new(name: EcoString, in_current_scope: bool, has_linkage: bool) -> Self {
        Self { name, in_current_scope, has_linkage }
    }
}

#[derive(Debug, Clone)]
pub struct Program(pub Vec<FuncDecl>);
impl Program {
    pub fn compile(&self) -> ir::Program {
        let mut buf = vec![];
        self.to_ir(&mut buf)
    }
    pub fn semantic_analysis(mut self) -> Option<Self> {
        // semantic analysis

        let mut id_map = Namespace::<IdCtx>::default();
        let mut symbols = Namespace::<Type>::default();

        for idx in 0..self.0.len() {
            self.0[idx] = self.0[idx]
                .clone()
                .resolve_switch_statements()?
                .resolve_identifiers(&mut id_map)?
                .type_check(&mut symbols)?
                .resolve_goto_labels()?
                .resolve_loop_labels()?;
        }

        Some(Self(self.0))
    }
}

#[derive(Debug, Clone, Copy)]
enum Type {
    Int,
    Func { arity: usize, defined: bool },
}

#[derive(Debug, Clone)]
pub enum Decl {
    Func(FuncDecl),
    Var(VarDecl),
}
impl Decl {
    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> Option<Self> {
        Some(match self {
            Self::Func(func) if func.body.is_none() => Self::Func(func.resolve_identifiers(map)?),
            Self::Func(_) => return None,
            Self::Var(var) => Self::Var(var.resolve_identifiers(map)?),
        })
    }

    fn type_check(self, symbols: &mut Namespace<Type>) -> Option<Self> {
        Some(match self {
            Self::Func(func) => Self::Func(func.type_check(symbols)?),
            Self::Var(var) => Self::Var(var.type_check(symbols)?),
        })
    }
}

#[derive(Debug, Clone)]
pub struct VarDecl {
    pub name: EcoString,
    pub init: Option<Expr>,
}
impl VarDecl {
    fn type_check(self, symbols: &mut Namespace<Type>) -> Option<Self> {
        symbols.insert(self.name.clone(), Type::Int);

        let init = match self.init {
            Some(exp) => Some(exp.type_check(symbols)?),
            None => None,
        };

        Some(Self { name: self.name, init })
    }

    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> Option<Self> {
        static DECL: AtomicUsize = AtomicUsize::new(0);

        if map.get(&self.name).is_some_and(|idctx| idctx.in_current_scope) {
            return None;
        }

        let unique_name = eco_format!("{}.{}", self.name, DECL.fetch_add(1, Relaxed));

        // shadowing happens here
        map.insert(self.name, IdCtx::new(unique_name.clone(), true, false));

        let init =
            if let Some(init) = self.init { Some(init.resolve_identifiers(map)?) } else { None };

        Some(VarDecl { name: unique_name, init })
    }
}

#[derive(Debug, Clone)]
pub struct FuncDecl {
    pub name: EcoString,
    pub params: Vec<EcoString>,
    pub body: Option<Block>,
}
impl FuncDecl {
    fn type_check(self, symbols: &mut Namespace<Type>) -> Option<Self> {
        let has_body = self.body.is_some();
        let mut already_defined = false;

        match symbols.get(&self.name) {
            None => {}
            Some(Type::Func { arity, defined }) if *arity == self.params.len() => {
                already_defined = *defined;
                if already_defined && has_body {
                    eprintln!("function defined multiple times.");
                    return None;
                }
            }
            _ => {
                eprintln!("incompatible function declarations.");
                return None;
            }
        }

        symbols.insert(
            self.name.clone(),
            Type::Func { arity: self.params.len(), defined: already_defined || has_body },
        );

        let body = if let Some(body) = self.body {
            for param in self.params.clone() {
                symbols.insert(param, Type::Int);
            }

            Some(body.type_check(symbols)?)
        } else {
            None
        };

        Some(Self { name: self.name, params: self.params, body })
    }

    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> Option<Self> {
        if map
            .insert(self.name.clone(), IdCtx::new(self.name.clone(), true, true))
            .is_some_and(|id| id.in_current_scope && !id.has_linkage)
        {
            return None;
        };

        let mut inner_map = map
            .iter()
            .map(|(k, idctx)| (k.clone(), IdCtx { in_current_scope: false, ..idctx.clone() }))
            .collect();

        let mut params = Vec::with_capacity(self.params.len());
        for param in self.params {
            let VarDecl { name, .. } = (VarDecl { name: param.clone(), init: None })
                .resolve_identifiers(&mut inner_map)?;

            params.push(name);
        }

        let body = match self.body {
            Some(body) => Some(body.resolve_identifiers(&mut inner_map)?),
            None => None,
        };

        Some(Self { name: self.name, params, body })
    }

    fn resolve_goto_labels(self) -> Option<Self> {
        // labels are function level
        let mut label_map = FxHashMap::default();
        let body = match self.body {
            Some(body) => Some(body.resolve_goto_labels(&mut label_map, self.name.clone())?),
            None => None,
        };

        if label_map.values().any(|v| !(*v)) {
            return None;
        };

        Some(Self { name: self.name, body, params: self.params })
    }

    fn resolve_loop_labels(self) -> Option<Self> {
        let body = match self.body {
            Some(body) => Some(body.resolve_loop_labels(LoopKind::None)?),
            None => None,
        };

        Some(Self { name: self.name, body, params: self.params })
    }

    fn resolve_switch_statements(self) -> Option<Self> {
        let body = match self.body {
            Some(body) => Some(body.resolve_switch_statements(None)?),
            None => None,
        };

        Some(Self { name: self.name, body, params: self.params })
    }
}

#[derive(Debug, Clone)]
pub enum BlockItem {
    S(Stmt),
    D(Decl),
}
impl BlockItem {
    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> Option<Self> {
        match self {
            Self::S(stmt) => Some(Self::S(stmt.resolve_identifiers(map)?)),
            Self::D(decl) => Some(Self::D(decl.resolve_identifiers(map)?)),
        }
    }

    fn resolve_goto_labels(self, labels: &mut Namespace<bool>, func_name: EcoString) -> Option<Self> {
        match self {
            Self::S(stmt) => Some(Self::S(stmt.resolve_goto_labels(labels, func_name)?)),
            d @ Self::D(_) => Some(d),
        }
    }

    fn resolve_loop_labels(self, current_label: LoopKind) -> Option<Self> {
        Some(match self {
            Self::S(stmt) => Self::S(stmt.resolve_loop_labels(current_label)?),
            Self::D(_) => self,
        })
    }

    fn resolve_switch_statements(self, in_switch: SwitchCtx) -> Option<Self> {
        Some(match self {
            Self::S(stmt) => Self::S(stmt.resolve_switch_statements(in_switch)?),
            Self::D(_) => self,
        })
    }

    fn type_check(self, symbols: &mut Namespace<Type>) -> Option<Self> {
        Some(match self {
            Self::S(stmt) => Self::S(stmt.type_check(symbols)?),
            Self::D(decl) => Self::D(decl.type_check(symbols)?),
        })
    }
}

type SwitchCtx<'s> = Option<&'s mut FxHashSet<Option<i32>>>;

#[derive(Debug, Clone)]
pub struct Block(pub Vec<BlockItem>);
impl Block {
    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> Option<Self> {
        let mut acc = Vec::with_capacity(self.0.len());

        for bi in self.0 {
            let bi = bi.resolve_identifiers(map)?;
            acc.push(bi);
        }

        Some(Self(acc))
    }
    fn resolve_goto_labels(self, labels: &mut Namespace<bool>, func_name: EcoString) -> Option<Self> {
        let mut acc = Vec::with_capacity(self.0.len());
        for bi in self.0 {
            let bi = bi.resolve_goto_labels(labels, func_name.clone())?;
            acc.push(bi);
        }

        Some(Self(acc))
    }

    fn resolve_loop_labels(self, current_label: LoopKind) -> Option<Self> {
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

    fn type_check(self, symbols: &mut Namespace<Type>) -> Option<Self> {
        let mut acc = Vec::with_capacity(self.0.len());

        for bi in self.0 {
            let bi = bi.type_check(symbols)?;
            acc.push(bi);
        }

        Some(Self(acc))
    }
}

#[derive(Debug, Clone)]
enum LoopKind {
    Loop(EcoString),
    Switch(EcoString),
    SwitchInLoop { loop_: EcoString, switch: EcoString },
    LoopInSwitch { loop_: EcoString, switch: EcoString },
    None,
}
impl LoopKind {
    fn break_label(&self) -> Option<EcoString> {
        match self {
            LoopKind::Loop(n)
            | LoopKind::Switch(n)
            | LoopKind::SwitchInLoop { switch: n, .. }
            | LoopKind::LoopInSwitch { loop_: n, .. } => Some(n.clone()),
            LoopKind::None => None,
        }
    }
    fn loop_label(&self) -> Option<EcoString> {
        match self {
            LoopKind::Loop(n)
            | LoopKind::SwitchInLoop { loop_: n, .. }
            | LoopKind::LoopInSwitch { loop_: n, .. } => Some(n.clone()),
            LoopKind::None | LoopKind::Switch(_) => None,
        }
    }
    fn switch_label(&self) -> Option<EcoString> {
        match self {
            LoopKind::Switch(n)
            | LoopKind::SwitchInLoop { switch: n, .. }
            | LoopKind::LoopInSwitch { switch: n, .. } => Some(n.clone()),
            LoopKind::None | LoopKind::Loop(_) => None,
        }
    }
    fn into_switch(self, label: EcoString) -> Self {
        match self {
            LoopKind::SwitchInLoop { loop_, .. }
            | LoopKind::LoopInSwitch { loop_, .. }
            | LoopKind::Loop(loop_) => LoopKind::SwitchInLoop { loop_, switch: label },
            LoopKind::Switch(_) | LoopKind::None => LoopKind::Switch(label),
        }
    }
    fn into_loop(self, label: EcoString) -> Self {
        match self {
            LoopKind::SwitchInLoop { switch, .. }
            | LoopKind::Switch(switch)
            | LoopKind::LoopInSwitch { switch, .. } => {
                LoopKind::LoopInSwitch { loop_: label, switch }
            }
            LoopKind::None | LoopKind::Loop(_) => LoopKind::Loop(label),
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
        init: Either<VarDecl, Option<Expr>>,
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
    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> Option<Self> {
        match self {
            Self::Return(expr) => Some(Self::Return(expr.resolve_identifiers(map)?)),
            Self::Expression(expr) => Some(Self::Expression(expr.resolve_identifiers(map)?)),
            Self::If { cond, then, else_ } => {
                let else_ = match else_ {
                    Some(s) => Some(Box::new(s.resolve_identifiers(map)?)),
                    None => None,
                };
                Some(Self::If {
                    cond: cond.resolve_identifiers(map)?,
                    then: Box::new(then.resolve_identifiers(map)?),
                    else_,
                })
            }

            Self::Label(name, stmt) => {
                Some(Self::Label(name, Box::new(stmt.resolve_identifiers(map)?)))
            }
            g @ Self::GoTo(_) => Some(g),

            Self::Compound(block) => {
                let mut block_map = map
                    .iter()
                    .map(|(k, idctx)| {
                        (k.clone(), IdCtx { in_current_scope: false, ..idctx.clone() })
                    })
                    .collect();

                Some(Self::Compound(block.resolve_identifiers(&mut block_map)?))
            }
            Self::While { cond, body, label } => {
                let cond = cond.resolve_identifiers(map)?;
                let body = Box::new(body.resolve_identifiers(map)?);

                Some(Self::While { cond, body, label: label.clone() })
            }
            Self::DoWhile { cond, body, label } => {
                let cond = cond.resolve_identifiers(map)?;
                let body = Box::new(body.resolve_identifiers(map)?);

                Some(Self::DoWhile { cond, body, label: label.clone() })
            }
            Self::For { init, cond, post, body, label } => {
                let mut for_map = map
                    .iter()
                    .map(|(k, idctx)| {
                        (k.clone(), IdCtx { in_current_scope: false, ..idctx.clone() })
                    })
                    .collect();

                let init = match init {
                    Left(decl) => Left(decl.resolve_identifiers(&mut for_map)?),
                    Right(Some(expr)) => Right(Some(expr.resolve_identifiers(&mut for_map)?)),
                    Right(None) => Right(None),
                };
                let cond = match cond {
                    Some(cond) => Some(cond.resolve_identifiers(&mut for_map)?),
                    None => None,
                };
                let post = match post {
                    Some(post) => Some(post.resolve_identifiers(&mut for_map)?),
                    None => None,
                };

                let body = Box::new(body.resolve_identifiers(&mut for_map)?);

                Some(Self::For { init, cond, post, body, label: label.clone() })
            }

            Self::Switch { ctrl, body, label, cases } => {
                let ctrl = ctrl.resolve_identifiers(map)?;
                let body = Box::new(body.resolve_identifiers(map)?);

                Some(Self::Switch { ctrl, body, label, cases })
            }
            Self::Case { cnst, body, label } => {
                let body = Box::new(body.resolve_identifiers(map)?);

                Some(Self::Case { cnst, body, label })
            }
            Self::Default { body, label } => {
                Some(Self::Default { body: Box::new(body.resolve_identifiers(map)?), label })
            }

            n @ (Self::Null | Self::Break(_) | Self::Continue(_)) => Some(n),
        }
    }

    fn resolve_goto_labels(self, labels: &mut Namespace<bool>, func_name: EcoString) -> Option<Self> {
        let mangle_label = |label| eco_format!("{func_name}.{}", label);
        match self {
            Self::GoTo(label) => {
                let label = mangle_label(label);
                if !labels.contains_key(&label) {
                    labels.insert(label.clone(), false);
                }

                Some(Self::GoTo(label))
            }
            Self::Label(label, s) => {
                let label = mangle_label(label);

                if labels.get(&label).is_some_and(|v| *v) {
                    eprintln!("repeated label : {label}");

                    return None;
                }
                labels.insert(label.clone(), true);
                let s = s.resolve_goto_labels(labels, func_name)?;

                Some(Self::Label(label, Box::new(s)))
            }
            Self::If { cond, then, else_ } => {
                let then = Box::new(then.resolve_goto_labels(labels, func_name.clone())?);
                let else_ = match else_ {
                    Some(s) => Some(Box::new(s.resolve_goto_labels(labels, func_name)?)),
                    None => None,
                };

                Some(Self::If { cond, then, else_ })
            }
            Self::Compound(block) => Some(Self::Compound(block.resolve_goto_labels(labels, func_name)?)),
            any @ (Self::Return(_)
            | Self::Null
            | Self::Expression(_)
            | Self::Break(_)
            | Self::Continue(_)) => Some(any),

            Self::While { body, cond, label } => {
                Some(Self::While { body: Box::new(body.resolve_goto_labels(labels, func_name)?), cond, label })
            }
            Self::DoWhile { body, cond, label } => Some(Self::DoWhile {
                body: Box::new(body.resolve_goto_labels(labels, func_name)?),
                cond,
                label,
            }),
            Self::For { init, cond, post, body, label } => Some(Self::For {
                init,
                cond,
                post,
                body: Box::new(body.resolve_goto_labels(labels, func_name)?),
                label,
            }),

            // extra extra credit // Incomplete
            Self::Switch { ctrl, body, label, cases } => Some(Self::Switch {
                ctrl,
                body: Box::new(body.resolve_goto_labels(labels, func_name)?),
                label,
                cases,
            }),
            Self::Case { cnst, body, label } => {
                Some(Self::Case { cnst, body: Box::new(body.resolve_goto_labels(labels, func_name)?), label })
            }
            Self::Default { body, label } => {
                Some(Self::Default { body: Box::new(body.resolve_goto_labels(labels, func_name)?), label })
            }
        }
    }

    fn resolve_loop_labels(self, current_label: LoopKind) -> Option<Self> {
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

    fn type_check(self, symbols: &mut Namespace<Type>) -> Option<Self> {
        Some(match self {
            Self::Return(expr) => Self::Return(expr.type_check(symbols)?),
            Self::Expression(expr) => Self::Expression(expr.type_check(symbols)?),
            Self::If { cond, then, else_ } => Self::If {
                cond: cond.type_check(symbols)?,
                then: Box::new(then.type_check(symbols)?),
                else_: match else_ {
                    Some(s) => Some(Box::new(s.type_check(symbols)?)),
                    None => None,
                },
            },
            Self::Compound(block) => Self::Compound(block.type_check(symbols)?),
            Self::While { cond, body, label } => Self::While {
                cond: cond.type_check(symbols)?,
                body: Box::new(body.type_check(symbols)?),
                label,
            },
            Self::DoWhile { body, cond, label } => Self::DoWhile {
                body: Box::new(body.type_check(symbols)?),
                cond: cond.type_check(symbols)?,
                label,
            },
            Self::For { init, cond, post, body, label } => Self::For {
                init: match init {
                    Left(d) => Left(d.type_check(symbols)?),
                    Right(Some(e)) => Right(Some(e.type_check(symbols)?)),
                    Right(None) => Right(None),
                },
                cond: match cond {
                    Some(e) => Some(e.type_check(symbols)?),
                    None => None,
                },
                post: match post {
                    Some(e) => Some(e.type_check(symbols)?),
                    None => None,
                },
                body: Box::new(body.type_check(symbols)?),
                label,
            },
            Self::Label(label, stmt) => Self::Label(label, Box::new(stmt.type_check(symbols)?)),
            Self::Switch { ctrl, body, label, cases } => Self::Switch {
                ctrl: ctrl.type_check(symbols)?,
                body: Box::new(body.type_check(symbols)?),
                label,
                cases,
            },
            Self::Case { cnst, body, label } => Self::Case {
                cnst: cnst.type_check(symbols)?,
                body: Box::new(body.type_check(symbols)?),
                label,
            },
            Self::Default { body, label } => {
                Self::Default { body: Box::new(body.type_check(symbols)?), label }
            }
            Self::GoTo(_) | Self::Break(_) | Self::Continue(_) | Self::Null => self,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    ConstInt(i32),
    Var(EcoString),
    Unary(UnaryOp, Box<Expr>),
    Binary { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    CompoundAssignment { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Assignemnt(Box<Expr>, Box<Expr>),
    Conditional { cond: Box<Expr>, then: Box<Expr>, else_: Box<Expr> },
    FuncCall { name: EcoString, args: Vec<Expr> },
}
impl Expr {
    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> Option<Self> {
        match self {
            Self::Assignemnt(left, right) if matches!(*left, Expr::Var(_)) => {
                Some(Self::Assignemnt(
                    Box::new(left.resolve_identifiers(map)?),
                    Box::new(right.resolve_identifiers(map)?),
                ))
            }
            Self::Assignemnt(_, _) => None,

            Self::Unary(op, expr) => {
                Some(Self::Unary(op, Box::new(expr.resolve_identifiers(map)?)))
            }

            Self::Binary { op, lhs, rhs } => Some(Self::Binary {
                op,
                lhs: Box::new(lhs.resolve_identifiers(map)?),
                rhs: Box::new(rhs.resolve_identifiers(map)?),
            }),

            Self::CompoundAssignment { op, lhs, rhs } => Some(Self::CompoundAssignment {
                op,
                lhs: Box::new(lhs.resolve_identifiers(map)?),
                rhs: Box::new(rhs.resolve_identifiers(map)?),
            }),

            v @ Self::ConstInt(_) => Some(v),
            Self::Conditional { cond, then, else_ } => Some(Self::Conditional {
                cond: Box::new(cond.resolve_identifiers(map)?),
                then: Box::new(then.resolve_identifiers(map)?),
                else_: Box::new(else_.resolve_identifiers(map)?),
            }),

            // magic happens here
            Self::Var(var) => map.get(&var).cloned().map(|t| Self::Var(t.name)),
            Self::FuncCall { name, args } => {
                let name = map.get(&name)?.clone().name;
                let mut resolved_args = Vec::with_capacity(args.len());

                for arg in args {
                    resolved_args.push(arg.resolve_identifiers(map)?);
                }

                Some(Self::FuncCall { name, args: resolved_args })
            }
        }
    }

    fn type_check(self, symbols: &mut Namespace<Type>) -> Option<Self> {
        match self {
            Expr::FuncCall { name, args } => {
                match symbols.get(name.as_str())? {
                    Type::Int => {
                        eprintln!("Variable used as function name");
                        return None;
                    }
                    Type::Func { arity, .. } if *arity != args.len() => {
                        eprintln!("function called with wrong number of arguments");
                        return None;
                    }
                    Type::Func { .. } => {}
                }

                for arg in args.clone() {
                    arg.type_check(symbols);
                }

                Some(Expr::FuncCall { name, args })
            }
            Expr::Var(name) => {
                if let Some(Type::Int) = symbols.get(name.as_str()) {
                    Some(Self::Var(name))
                } else {
                    eprintln!("function used as variable");
                    None
                }
            }

            // --
            Expr::Unary(op, expr) => Some(Self::Unary(op, Box::new(expr.type_check(symbols)?))),

            Expr::Binary { op, lhs, rhs } => Some(Self::Binary {
                op,
                lhs: Box::new(lhs.type_check(symbols)?),
                rhs: Box::new(rhs.type_check(symbols)?),
            }),

            Expr::Assignemnt(lhs, rhs) => Some(Self::Assignemnt(
                Box::new(lhs.type_check(symbols)?),
                Box::new(rhs.type_check(symbols)?),
            )),

            Expr::CompoundAssignment { op, lhs, rhs } => Some(Self::CompoundAssignment {
                op,
                lhs: Box::new(lhs.type_check(symbols)?),
                rhs: Box::new(rhs.type_check(symbols)?),
            }),
            Expr::ConstInt(_) => Some(self),
            Expr::Conditional { cond, then, else_ } => Some(Self::Conditional {
                cond: Box::new(cond.type_check(symbols)?),
                then: Box::new(then.type_check(symbols)?),
                else_: Box::new(else_.type_check(symbols)?),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

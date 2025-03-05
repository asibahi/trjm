use crate::ir::{self, GEN, ToIr};
use ecow::{EcoString as Ecow, eco_format};
use either::Either::{self, Left, Right};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    collections::{HashSet, hash_map::Entry},
    sync::atomic::Ordering::Relaxed,
};

pub type Namespace<T> = FxHashMap<Ecow, T>;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct IdCtx {
    name: Ecow,
    in_current_scope: bool,
    has_linkage: bool,
}
impl IdCtx {
    fn new(name: Ecow, in_current_scope: bool, has_linkage: bool) -> Self {
        Self { name, in_current_scope, has_linkage }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeCtx {
    pub type_: Type,
    pub attr: Attributes,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub decls: Vec<Decl>,
    pub symbols: Namespace<TypeCtx>,
}
impl Program {
    pub fn new(decls: Vec<Decl>) -> Self {
        Self { decls, symbols: Namespace::default() }
    }
    pub fn compile(&self) -> ir::Program {
        let mut buf = vec![];
        self.to_ir(&mut buf)
    }
    pub fn semantic_analysis(mut self) -> anyhow::Result<Self> {
        // semantic analysis

        let mut id_map = Namespace::<IdCtx>::default();

        for idx in 0..self.decls.len() {
            self.decls[idx] = self.decls[idx]
                .clone()
                .resolve_switch_statements()?
                .resolve_identifiers(Scope::File, &mut id_map)?
                .type_check(Scope::File, &mut self.symbols)?
                .resolve_goto_labels()?
                .resolve_loop_labels()?;
        }

        Ok(Self { decls: self.decls, symbols: self.symbols })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Long,
    Func { params: Vec<Type>, ret: Box<Type> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attributes {
    Fun { defined: bool, global: bool },
    Static { init: InitValue, global: bool },
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitValue {
    Tentative,
    Initial(i32),
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    Static,
    Extern,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    File,
    Block,
}

#[derive(Debug, Clone)]
pub enum Decl {
    Func(FuncDecl),
    Var(VarDecl),
}

impl Decl {
    fn resolve_identifiers(
        self,
        current_scope: Scope,
        map: &mut Namespace<IdCtx>,
    ) -> anyhow::Result<Self> {
        Ok(match (self, current_scope) {
            (Self::Func(func), Scope::Block) if func.body.is_some() => {
                anyhow::bail!("function definition in block scope");
            }
            (Self::Func(func), Scope::Block) if func.sc == StorageClass::Static => {
                anyhow::bail!("can't have static functions in block scope");
            }
            (Self::Func(func), _) => Self::Func(func.resolve_identifiers(map)?),
            (Self::Var(var), Scope::Block) => Self::Var(var.resolve_identifiers(map)?),
            (Self::Var(var), Scope::File) => {
                map.insert(
                    var.clone().name,
                    IdCtx { name: var.clone().name, in_current_scope: true, has_linkage: true },
                );
                Self::Var(var)
            }
        })
    }

    fn type_check(
        self,
        current_scope: Scope,
        symbols: &mut Namespace<TypeCtx>,
    ) -> anyhow::Result<Self> {
        Ok(match (self, current_scope) {
            (Self::Func(func), _) => Self::Func(func.type_check(symbols)?),
            (Self::Var(var), Scope::Block) => Self::Var(var.type_check_block(symbols)?),
            (Self::Var(var), Scope::File) => Self::Var(var.type_check_file(symbols)?),
        })
    }

    fn resolve_switch_statements(self) -> anyhow::Result<Self> {
        Ok(match self {
            Decl::Func(func_decl) => Self::Func(func_decl.resolve_switch_statements()?),
            Decl::Var(_) => self,
        })
    }

    fn resolve_goto_labels(self) -> anyhow::Result<Self> {
        Ok(match self {
            Decl::Func(func_decl) => Self::Func(func_decl.resolve_goto_labels()?),
            Decl::Var(_) => self,
        })
    }

    fn resolve_loop_labels(self) -> anyhow::Result<Self> {
        Ok(match self {
            Decl::Func(func_decl) => Self::Func(func_decl.resolve_loop_labels()?),
            Decl::Var(_) => self,
        })
    }
}

#[derive(Debug, Clone)]
pub struct VarDecl {
    pub name: Ecow,
    pub init: Option<Expr>,
    pub sc: StorageClass,
    pub var_type: Type,
}
impl VarDecl {
    fn type_check_file(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        let mut init = match self.init {
            Some(Expr::Const(Const::Int(i))) => InitValue::Initial(i),
            None if self.sc == StorageClass::Extern => InitValue::None,
            None => InitValue::Tentative,
            _ => anyhow::bail!("non-constant initializer"),
        };

        let mut global = self.sc != StorageClass::Static;

        match symbols.get(&self.name) {
            None => {}
            Some(TypeCtx { type_: Type::Func { .. }, .. }) => {
                // not an int
                anyhow::bail!("function redclared as variable");
            }
            Some(TypeCtx { attr: Attributes::Local | Attributes::Fun { .. }, .. }) => {
                unreachable!()
            }
            Some(TypeCtx { attr: Attributes::Static { init: o_i, global: o_g }, .. }) => {
                match self.sc {
                    StorageClass::Extern => global = *o_g,
                    _ if *o_g != global => anyhow::bail!("Conflicting variable linkage"),

                    _ => {}
                }

                match (init, *o_i) {
                    (InitValue::Initial(_), InitValue::Initial(_)) => {
                        anyhow::bail!("Conflicting file scope definitions");
                    }

                    (InitValue::Initial(i), _) | (_, InitValue::Initial(i)) => {
                        init = InitValue::Initial(i);
                    }

                    (InitValue::Tentative, _) | (_, InitValue::Tentative) => {
                        init = InitValue::Tentative;
                    }

                    (InitValue::None, InitValue::None) => {}
                };
            }
        }

        let attrs = Attributes::Static { init, global };
        symbols.insert(self.name.clone(), TypeCtx { type_: Type::Int, attr: attrs });

        Ok(self)
    }
    fn type_check_block(mut self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        match self.sc {
            StorageClass::Extern if self.init.is_some() => {
                anyhow::bail!("init on local extern variable");
            }
            StorageClass::Extern => match symbols.entry(self.name.clone()) {
                Entry::Occupied(occ) if occ.get().type_ != Type::Int => {
                    anyhow::bail!("function redeclared as variable");
                }
                Entry::Vacant(vac) => {
                    vac.insert(TypeCtx {
                        type_: Type::Int,
                        attr: Attributes::Static { init: InitValue::None, global: true },
                    });
                }
                Entry::Occupied(_) => {}
            },
            StorageClass::Static => {
                let init_value = match self.init.take() {
                    Some(Expr::Const(Const::Int(i))) => InitValue::Initial(i),
                    None => InitValue::Initial(0),
                    _ => {
                        anyhow::bail!("non-constant initializer on local static variable");
                    }
                };

                symbols.insert(
                    self.name.clone(),
                    TypeCtx {
                        type_: Type::Int,
                        attr: Attributes::Static { init: init_value, global: false },
                    },
                );
            }
            StorageClass::None => {
                symbols.insert(
                    self.name.clone(),
                    TypeCtx { type_: Type::Int, attr: Attributes::Local },
                );

                self.init = match self.init {
                    Some(v) => Some(v.type_check(symbols)?),
                    None => None,
                };
            }
        }

        Ok(Self { name: self.name, init: self.init, sc: self.sc, var_type: todo!() })
    }

    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> anyhow::Result<Self> {
        if map.get(&self.name).is_some_and(|idctx| {
            idctx.in_current_scope && !(idctx.has_linkage && self.sc == StorageClass::Extern)
        }) {
            anyhow::bail!("conflicting local declarations");
        }

        let (name, sc) = match self.sc {
            StorageClass::Extern if self.init.is_some() => {
                anyhow::bail!("extern declarations with initial value.")
            }
            StorageClass::Extern => {
                map.insert(self.name.clone(), IdCtx::new(self.name.clone(), true, true));
                (self.name, StorageClass::Extern)
            }
            sc => {
                let unique_name = eco_format!("{}.{}", self.name, GEN.fetch_add(1, Relaxed));

                // shadowing happens here
                map.insert(self.name, IdCtx::new(unique_name.clone(), true, false));

                (unique_name, sc)
            }
        };

        let init =
            if let Some(init) = self.init { Some(init.resolve_identifiers(map)?) } else { None };

        Ok(VarDecl { name, init, sc, var_type: todo!() })
    }
}

#[derive(Debug, Clone)]
pub struct FuncDecl {
    pub name: Ecow,
    pub params: Vec<Ecow>,
    pub body: Option<Block>,
    pub sc: StorageClass,
    pub fun_type: Type,
}
impl FuncDecl {
    fn type_check(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        let has_body = self.body.is_some();
        let mut already_defined = false;
        let mut global = self.sc != StorageClass::Static;

        match symbols.get(&self.name) {
            None => {}
            Some(TypeCtx { type_: Type::Func { .. }, attr: Attributes::Fun { defined, .. } })
                if *defined && has_body =>
            {
                anyhow::bail!("multiple function definitions")
            }

            Some(TypeCtx { type_: Type::Func { .. }, attr: Attributes::Fun { global, .. } })
                if *global && self.sc == StorageClass::Static =>
            {
                anyhow::bail!("Static function declaration follows non-static")
            }

            Some(TypeCtx {
                type_: Type::Func { params, ret },
                attr: Attributes::Fun { defined, global: old_global },
            }) if params.len() == self.params.len() => {
                already_defined = *defined;
                global = *old_global;
            }
            _ => anyhow::bail!("incompatible function declarations"),
        }

        symbols.insert(
            self.name.clone(),
            TypeCtx {
                type_: Type::Func { params: todo!(), ret: todo!() },
                attr: Attributes::Fun { defined: already_defined || has_body, global },
            },
        );

        let body = if let Some(body) = self.body {
            for param in self.params.clone() {
                symbols.insert(param, TypeCtx { type_: Type::Int, attr: Attributes::Local });
            }

            Some(body.type_check(symbols)?)
        } else {
            None
        };

        Ok(Self {
            name: self.name,
            params: self.params,
            body,
            sc: if global { StorageClass::Extern } else { StorageClass::Static },
            fun_type: todo!(),
        })
    }

    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> anyhow::Result<Self> {
        if map
            .insert(self.name.clone(), IdCtx::new(self.name.clone(), true, true))
            .is_some_and(|id| id.in_current_scope && !id.has_linkage)
        {
            anyhow::bail!("duplicate declaration. listing 9-19");
        };

        let mut inner_map = map
            .iter()
            .map(|(k, idctx)| (k.clone(), IdCtx { in_current_scope: false, ..idctx.clone() }))
            .collect();

        let mut params = Vec::with_capacity(self.params.len());
        for param in self.params {
            let VarDecl { name, .. } = (VarDecl {
                name: param.clone(),
                init: None,
                sc: StorageClass::None,
                var_type: todo!(),
            })
            .resolve_identifiers(&mut inner_map)?;

            params.push(name);
        }

        let body = match self.body {
            Some(body) => Some(body.resolve_identifiers(&mut inner_map)?),
            None => None,
        };

        Ok(Self { name: self.name, params, body, sc: self.sc, fun_type: todo!() })
    }

    fn resolve_goto_labels(self) -> anyhow::Result<Self> {
        // labels are function level
        let mut label_map = FxHashMap::default();
        let body = match self.body {
            Some(body) => Some(body.resolve_goto_labels(&mut label_map, self.name.clone())?),
            None => None,
        };

        if label_map.values().any(|v| !(*v)) {
            anyhow::bail!("goto label already exists in function");
        };

        Ok(Self { name: self.name, body, params: self.params, sc: self.sc, fun_type: todo!() })
    }

    fn resolve_loop_labels(self) -> anyhow::Result<Self> {
        let body = match self.body {
            Some(body) => Some(body.resolve_loop_labels(LoopKind::None)?),
            None => None,
        };

        Ok(Self { name: self.name, body, params: self.params, sc: self.sc, fun_type: todo!() })
    }

    fn resolve_switch_statements(self) -> anyhow::Result<Self> {
        let body = match self.body {
            Some(body) => Some(body.resolve_switch_statements(None)?),
            None => None,
        };

        Ok(Self { name: self.name, body, params: self.params, sc: self.sc, fun_type: todo!() })
    }
}

#[derive(Debug, Clone)]
pub enum BlockItem {
    S(Stmt),
    D(Decl),
}
impl BlockItem {
    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> anyhow::Result<Self> {
        match self {
            Self::S(stmt) => Ok(Self::S(stmt.resolve_identifiers(map)?)),
            Self::D(decl) => Ok(Self::D(decl.resolve_identifiers(Scope::Block, map)?)),
        }
    }

    fn resolve_goto_labels(
        self,
        labels: &mut Namespace<bool>,
        func_name: Ecow,
    ) -> anyhow::Result<Self> {
        match self {
            Self::S(stmt) => Ok(Self::S(stmt.resolve_goto_labels(labels, func_name)?)),
            d @ Self::D(_) => Ok(d),
        }
    }

    fn resolve_loop_labels(self, current_label: LoopKind) -> anyhow::Result<Self> {
        Ok(match self {
            Self::S(stmt) => Self::S(stmt.resolve_loop_labels(current_label)?),
            Self::D(_) => self,
        })
    }

    fn resolve_switch_statements(self, in_switch: SwitchCtx) -> anyhow::Result<Self> {
        Ok(match self {
            Self::S(stmt) => Self::S(stmt.resolve_switch_statements(in_switch)?),
            Self::D(_) => self,
        })
    }

    fn type_check(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        Ok(match self {
            Self::S(stmt) => Self::S(stmt.type_check(symbols)?),
            Self::D(decl) => Self::D(decl.type_check(Scope::Block, symbols)?),
        })
    }
}

type SwitchCtx<'s> = Option<&'s mut FxHashSet<Option<i32>>>;

#[derive(Debug, Clone)]
pub struct Block(pub Vec<BlockItem>);
impl Block {
    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> anyhow::Result<Self> {
        let mut acc = Vec::with_capacity(self.0.len());

        for bi in self.0 {
            let bi = bi.resolve_identifiers(map)?;
            acc.push(bi);
        }

        Ok(Self(acc))
    }
    fn resolve_goto_labels(
        self,
        labels: &mut Namespace<bool>,
        func_name: Ecow,
    ) -> anyhow::Result<Self> {
        let mut acc = Vec::with_capacity(self.0.len());
        for bi in self.0 {
            let bi = bi.resolve_goto_labels(labels, func_name.clone())?;
            acc.push(bi);
        }

        Ok(Self(acc))
    }

    fn resolve_loop_labels(self, current_label: LoopKind) -> anyhow::Result<Self> {
        let mut acc = Vec::with_capacity(self.0.len());
        let current_label = current_label; // pedantic clippy

        for bi in self.0 {
            let bi = bi.resolve_loop_labels(current_label.clone())?;
            acc.push(bi);
        }

        Ok(Self(acc))
    }

    fn resolve_switch_statements(self, switch_ctx: SwitchCtx) -> anyhow::Result<Self> {
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

        Ok(Self(acc))
    }

    fn type_check(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        let mut acc = Vec::with_capacity(self.0.len());

        for bi in self.0 {
            let bi = bi.type_check(symbols)?;
            acc.push(bi);
        }

        Ok(Self(acc))
    }
}

#[derive(Debug, Clone)]
enum LoopKind {
    Loop(Ecow),
    Switch(Ecow),
    SwitchInLoop { loop_: Ecow, switch: Ecow },
    LoopInSwitch { loop_: Ecow, switch: Ecow },
    None,
}
impl LoopKind {
    fn break_label(&self) -> Option<Ecow> {
        match self {
            LoopKind::Loop(n)
            | LoopKind::Switch(n)
            | LoopKind::SwitchInLoop { switch: n, .. }
            | LoopKind::LoopInSwitch { loop_: n, .. } => Some(n.clone()),
            LoopKind::None => None,
        }
    }
    fn loop_label(&self) -> Option<Ecow> {
        match self {
            LoopKind::Loop(n)
            | LoopKind::SwitchInLoop { loop_: n, .. }
            | LoopKind::LoopInSwitch { loop_: n, .. } => Some(n.clone()),
            LoopKind::None | LoopKind::Switch(_) => None,
        }
    }
    fn switch_label(&self) -> Option<Ecow> {
        match self {
            LoopKind::Switch(n)
            | LoopKind::SwitchInLoop { switch: n, .. }
            | LoopKind::LoopInSwitch { switch: n, .. } => Some(n.clone()),
            LoopKind::None | LoopKind::Loop(_) => None,
        }
    }
    fn into_switch(self, label: Ecow) -> Self {
        match self {
            LoopKind::SwitchInLoop { loop_, .. }
            | LoopKind::LoopInSwitch { loop_, .. }
            | LoopKind::Loop(loop_) => LoopKind::SwitchInLoop { loop_, switch: label },
            LoopKind::Switch(_) | LoopKind::None => LoopKind::Switch(label),
        }
    }
    fn into_loop(self, label: Ecow) -> Self {
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

    Break(Option<Ecow>),
    Continue(Option<Ecow>),
    While {
        cond: Expr,
        body: Box<Stmt>,
        label: Option<Ecow>,
    },
    DoWhile {
        body: Box<Stmt>,
        cond: Expr,
        label: Option<Ecow>,
    },
    For {
        init: Either<VarDecl, Option<Expr>>,
        cond: Option<Expr>,
        post: Option<Expr>,

        body: Box<Stmt>,
        label: Option<Ecow>,
    },

    // extra credit
    GoTo(Ecow),
    Label(Ecow, Box<Stmt>),

    Switch {
        ctrl: Expr,
        body: Box<Stmt>,
        label: Option<Ecow>,
        cases: Vec<Option<i32>>,
    },
    Case {
        cnst: Expr,
        body: Box<Stmt>,
        label: Option<Ecow>,
    },
    Default {
        body: Box<Stmt>,
        label: Option<Ecow>,
    },

    Null,
}
impl Stmt {
    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> anyhow::Result<Self> {
        match self {
            Self::Return(expr) => Ok(Self::Return(expr.resolve_identifiers(map)?)),
            Self::Expression(expr) => Ok(Self::Expression(expr.resolve_identifiers(map)?)),
            Self::If { cond, then, else_ } => {
                let else_ = match else_ {
                    Some(s) => Some(Box::new(s.resolve_identifiers(map)?)),
                    None => None,
                };
                Ok(Self::If {
                    cond: cond.resolve_identifiers(map)?,
                    then: Box::new(then.resolve_identifiers(map)?),
                    else_,
                })
            }

            Self::Label(name, stmt) => {
                Ok(Self::Label(name, Box::new(stmt.resolve_identifiers(map)?)))
            }
            g @ Self::GoTo(_) => Ok(g),

            Self::Compound(block) => {
                let mut block_map = map
                    .iter()
                    .map(|(k, idctx)| {
                        (k.clone(), IdCtx { in_current_scope: false, ..idctx.clone() })
                    })
                    .collect();

                Ok(Self::Compound(block.resolve_identifiers(&mut block_map)?))
            }
            Self::While { cond, body, label } => {
                let cond = cond.resolve_identifiers(map)?;
                let body = Box::new(body.resolve_identifiers(map)?);

                Ok(Self::While { cond, body, label: label.clone() })
            }
            Self::DoWhile { cond, body, label } => {
                let cond = cond.resolve_identifiers(map)?;
                let body = Box::new(body.resolve_identifiers(map)?);

                Ok(Self::DoWhile { cond, body, label: label.clone() })
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

                Ok(Self::For { init, cond, post, body, label: label.clone() })
            }

            Self::Switch { ctrl, body, label, cases } => {
                let ctrl = ctrl.resolve_identifiers(map)?;
                let body = Box::new(body.resolve_identifiers(map)?);

                Ok(Self::Switch { ctrl, body, label, cases })
            }
            Self::Case { cnst, body, label } => {
                let body = Box::new(body.resolve_identifiers(map)?);

                Ok(Self::Case { cnst, body, label })
            }
            Self::Default { body, label } => {
                Ok(Self::Default { body: Box::new(body.resolve_identifiers(map)?), label })
            }

            n @ (Self::Null | Self::Break(_) | Self::Continue(_)) => Ok(n),
        }
    }

    fn resolve_goto_labels(
        self,
        labels: &mut Namespace<bool>,
        func_name: Ecow,
    ) -> anyhow::Result<Self> {
        let mangle_label = |label| eco_format!("{func_name}.{}", label);
        match self {
            Self::GoTo(label) => {
                let label = mangle_label(label);
                if !labels.contains_key(&label) {
                    labels.insert(label.clone(), false);
                }

                Ok(Self::GoTo(label))
            }
            Self::Label(label, s) => {
                let label = mangle_label(label);

                if labels.get(&label).is_some_and(|v| *v) {
                    anyhow::bail!("repeated label : {label}");
                }
                labels.insert(label.clone(), true);
                let s = s.resolve_goto_labels(labels, func_name)?;

                Ok(Self::Label(label, Box::new(s)))
            }
            Self::If { cond, then, else_ } => {
                let then = Box::new(then.resolve_goto_labels(labels, func_name.clone())?);
                let else_ = match else_ {
                    Some(s) => Some(Box::new(s.resolve_goto_labels(labels, func_name)?)),
                    None => None,
                };

                Ok(Self::If { cond, then, else_ })
            }
            Self::Compound(block) => {
                Ok(Self::Compound(block.resolve_goto_labels(labels, func_name)?))
            }
            any @ (Self::Return(_)
            | Self::Null
            | Self::Expression(_)
            | Self::Break(_)
            | Self::Continue(_)) => Ok(any),

            Self::While { body, cond, label } => Ok(Self::While {
                body: Box::new(body.resolve_goto_labels(labels, func_name)?),
                cond,
                label,
            }),
            Self::DoWhile { body, cond, label } => Ok(Self::DoWhile {
                body: Box::new(body.resolve_goto_labels(labels, func_name)?),
                cond,
                label,
            }),
            Self::For { init, cond, post, body, label } => Ok(Self::For {
                init,
                cond,
                post,
                body: Box::new(body.resolve_goto_labels(labels, func_name)?),
                label,
            }),

            // extra extra credit // Incomplete
            Self::Switch { ctrl, body, label, cases } => Ok(Self::Switch {
                ctrl,
                body: Box::new(body.resolve_goto_labels(labels, func_name)?),
                label,
                cases,
            }),
            Self::Case { cnst, body, label } => Ok(Self::Case {
                cnst,
                body: Box::new(body.resolve_goto_labels(labels, func_name)?),
                label,
            }),
            Self::Default { body, label } => Ok(Self::Default {
                body: Box::new(body.resolve_goto_labels(labels, func_name)?),
                label,
            }),
        }
    }

    fn resolve_loop_labels(self, current_label: LoopKind) -> anyhow::Result<Self> {
        let loop_counter = GEN.fetch_add(1, Relaxed);

        match self {
            Self::Break(_) => Ok(Self::Break(Some(
                current_label.break_label().ok_or(anyhow::anyhow!("not a break label"))?,
            ))),
            Self::Continue(_) => Ok(Self::Continue(Some(
                current_label.loop_label().ok_or(anyhow::anyhow!("not a continue label"))?,
            ))),

            Self::While { cond, body, .. } => {
                let new_label = current_label.into_loop(eco_format!("while.{}", loop_counter));
                let body = Box::new(body.resolve_loop_labels(new_label.clone())?);

                Ok(Self::While { cond, body, label: new_label.break_label() })
            }
            Self::DoWhile { cond, body, .. } => {
                let new_label = current_label.into_loop(eco_format!("do.{}", loop_counter));
                let body = Box::new(body.resolve_loop_labels(new_label.clone())?);

                Ok(Self::DoWhile { cond, body, label: new_label.break_label() })
            }
            Self::For { init, cond, post, body, .. } => {
                let new_label = current_label.into_loop(eco_format!("for.{}", loop_counter));
                let body = Box::new(body.resolve_loop_labels(new_label.clone())?);

                Ok(Self::For { init, cond, post, body, label: new_label.break_label() })
            }
            Self::Switch { ctrl, body, cases, .. } => {
                let new_label = current_label.into_switch(eco_format!("swch.{}", loop_counter));

                let body = Box::new(body.resolve_loop_labels(new_label.clone())?);

                Ok(Self::Switch { ctrl, body, label: new_label.break_label(), cases })
            }
            Self::Case { cnst, body, .. } => Ok(Self::Case {
                cnst,
                body: Box::new(body.resolve_loop_labels(current_label.clone())?),
                label: current_label.switch_label(),
            }),
            Self::Default { body, .. } => Ok(Self::Default {
                body: Box::new(body.resolve_loop_labels(current_label.clone())?),
                label: current_label.switch_label(),
            }),

            Self::If { cond, then, else_ } => {
                let then = Box::new(then.resolve_loop_labels(current_label.clone())?);
                let else_ = match else_ {
                    Some(stmt) => Some(Box::new(stmt.resolve_loop_labels(current_label)?)),
                    None => None,
                };

                Ok(Self::If { cond, then, else_ })
            }
            Self::Compound(block) => Ok(Self::Compound(block.resolve_loop_labels(current_label)?)),
            Self::Label(label, stmt) => {
                let stmt = Box::new(stmt.resolve_loop_labels(current_label)?);
                Ok(Self::Label(label, stmt))
            }

            Self::Return(_) | Self::Expression(_) | Self::GoTo(_) | Self::Null => Ok(self),
        }
    }

    fn resolve_switch_statements(self, switch_ctx: SwitchCtx) -> anyhow::Result<Self> {
        Ok(match self {
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
                let switch_ctx = switch_ctx.ok_or(anyhow::anyhow!("case outside of a switch"))?;

                let Expr::Const(Const::Int(value)) = cnst else {
                    anyhow::bail!("case value not a constant");
                };

                if !switch_ctx.insert(Some(value)) {
                    anyhow::bail!("case value already exists");
                };

                let body = Box::new(body.resolve_switch_statements(Some(switch_ctx))?);
                Self::Case { cnst, body, label }
            }
            Self::Default { body, label } => {
                let switch_ctx =
                    switch_ctx.ok_or(anyhow::anyhow!("default outside of a switch"))?;

                if !switch_ctx.insert(None) {
                    anyhow::bail!("cdefault ase already exists");
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

    fn type_check(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        Ok(match self {
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
                    Left(e) if e.sc != StorageClass::None => {
                        anyhow::bail!("specifier in for loop");
                    }
                    Left(d) => Left(d.type_check_block(symbols)?),
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
    Const(Const),
    Var(Ecow),
    Cast { to: Type, from: Box<Expr> },
    Unary(UnaryOp, Box<Expr>),
    Binary { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    CompoundAssignment { op: BinaryOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Assignemnt(Box<Expr>, Box<Expr>),
    Conditional { cond: Box<Expr>, then: Box<Expr>, else_: Box<Expr> },
    FuncCall { name: Ecow, args: Vec<Expr> },
}
impl Expr {
    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> anyhow::Result<Self> {
        match self {
            Self::Assignemnt(left, right) if matches!(*left, Expr::Var(_)) => Ok(Self::Assignemnt(
                Box::new(left.resolve_identifiers(map)?),
                Box::new(right.resolve_identifiers(map)?),
            )),
            Self::Assignemnt(_, _) => anyhow::bail!("left value isn't a variable"),

            Self::Unary(op, expr) => Ok(Self::Unary(op, Box::new(expr.resolve_identifiers(map)?))),

            Self::Binary { op, lhs, rhs } => Ok(Self::Binary {
                op,
                lhs: Box::new(lhs.resolve_identifiers(map)?),
                rhs: Box::new(rhs.resolve_identifiers(map)?),
            }),

            Self::CompoundAssignment { op, lhs, rhs } => Ok(Self::CompoundAssignment {
                op,
                lhs: Box::new(lhs.resolve_identifiers(map)?),
                rhs: Box::new(rhs.resolve_identifiers(map)?),
            }),

            v @ Self::Const(_) => Ok(v),
            Self::Conditional { cond, then, else_ } => Ok(Self::Conditional {
                cond: Box::new(cond.resolve_identifiers(map)?),
                then: Box::new(then.resolve_identifiers(map)?),
                else_: Box::new(else_.resolve_identifiers(map)?),
            }),

            // magic happens here
            Self::Var(var) => map
                .get(&var)
                .cloned()
                .map(|t| Self::Var(t.name))
                .ok_or(anyhow::anyhow!("variable does not exist in map")),
            Self::FuncCall { name, args } => {
                let name = map
                    .get(&name)
                    .ok_or(anyhow::anyhow!("function does not exist in map"))?
                    .clone()
                    .name;
                let mut resolved_args = Vec::with_capacity(args.len());

                for arg in args {
                    resolved_args.push(arg.resolve_identifiers(map)?);
                }

                Ok(Self::FuncCall { name, args: resolved_args })
            }
            Self::Cast { .. } => todo!(),
        }
    }

    fn type_check(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        match self {
            Self::FuncCall { name, args } => {
                match &symbols
                    .get(name.as_str())
                    .ok_or(anyhow::anyhow!("function does not exist in symbol map"))?
                    .type_
                {
                    Type::Func { params, .. } if params.len() != args.len() => {
                        anyhow::bail!("function called with wrong number of arguments");
                    }
                    Type::Func { .. } => {}
                    _ => {
                        anyhow::bail!("Variable used as function name");
                    }
                }

                for arg in args.clone() {
                    arg.type_check(symbols)?;
                }

                Ok(Self::FuncCall { name, args })
            }
            Self::Var(name) => {
                if let Some(Type::Int) = symbols.get(name.as_str()).map(|c| c.type_.clone()) {
                    Ok(Self::Var(name))
                } else {
                    anyhow::bail!("function used as variable");
                }
            }

            // --
            Self::Unary(op, expr) => Ok(Self::Unary(op, Box::new(expr.type_check(symbols)?))),

            Self::Binary { op, lhs, rhs } => Ok(Self::Binary {
                op,
                lhs: Box::new(lhs.type_check(symbols)?),
                rhs: Box::new(rhs.type_check(symbols)?),
            }),

            Self::Assignemnt(lhs, rhs) => Ok(Self::Assignemnt(
                Box::new(lhs.type_check(symbols)?),
                Box::new(rhs.type_check(symbols)?),
            )),

            Self::CompoundAssignment { op, lhs, rhs } => Ok(Self::CompoundAssignment {
                op,
                lhs: Box::new(lhs.type_check(symbols)?),
                rhs: Box::new(rhs.type_check(symbols)?),
            }),
            Self::Const(_) => Ok(self),
            Self::Conditional { cond, then, else_ } => Ok(Self::Conditional {
                cond: Box::new(cond.type_check(symbols)?),
                then: Box::new(then.type_check(symbols)?),
                else_: Box::new(else_.type_check(symbols)?),
            }),

            Self::Cast { .. } => todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Const {
    Int(i32),
    Long(i64),
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

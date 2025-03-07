use crate::ir::{self, GEN};
use ecow::{EcoString as Ecow, eco_format};
use either::Either::{self, Left, Right};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    collections::{HashSet, hash_map::Entry},
    fmt::Write,
    ops::Deref,
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
impl std::fmt::Display for TypeCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let attr = match self.attr {
            Func { defined: false, global: true } => "global declaration".into(),
            Func { defined: false, global: false } => "local declaration".into(),
            Static { init, global: true } => eco_format!("global {init}"),
            Static { init, global: false } => eco_format!("inner  {init}"),
            Local => "local".into(),
            _ => "".into(),
        };

        write!(f, "{:<8} {attr}", self.type_)
    }
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

    pub fn semantic_analysis(mut self) -> anyhow::Result<Self> {
        // semantic analysis

        let mut id_map = Namespace::<IdCtx>::default();

        for idx in 0..self.decls.len() {
            self.decls[idx] = self.decls[idx]
                .clone()
                .resolve_identifiers(Scope::File, &mut id_map)?
                .type_check(Scope::File, &mut self.symbols)?
                .resolve_switch_statements()?
                .resolve_goto_labels()?
                .resolve_loop_labels()?;
        }

        Ok(Self { decls: self.decls, symbols: self.symbols })
    }

    pub fn compile(mut self) -> ir::Program {
        let mut buf = vec![];
        let mut symbols = std::mem::take(&mut self.symbols);

        self.to_ir(&mut buf, &mut symbols)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Long,
    Func { params: Vec<Type>, ret: Box<Type> },
}
impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int => f.pad("int"),
            Type::Long => f.pad("long"),
            Type::Func { params, ret } => {
                let mut buf = Ecow::new();
                write!(buf, "func (")?;
                for param in params {
                    write!(buf, "{param}, ")?;
                }
                write!(buf, ") -> {ret}")?;

                f.pad(&buf)
            }
        }
    }
}

impl Type {
    fn get_common_type(self, other: Self) -> Self {
        // if either of these is a function shouldn't we panic ?
        // why is function in the same type anyway ?
        if self == other { self } else { Type::Long }
    }
    pub fn zeroed_static(&self) -> StaticInit {
        match self {
            Type::Int => StaticInit::Int(0),
            Type::Long => StaticInit::Long(0),
            Type::Func { .. } => unreachable!("function static value not a thing"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attributes {
    Func { defined: bool, global: bool },
    Static { init: InitValue, global: bool },
    Local,
}
pub use Attributes::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitValue {
    Tentative,
    Initial(StaticInit),
    NoInit,
}
pub use InitValue::*;
impl std::fmt::Display for InitValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tentative => write!(f, "tentative"),
            Initial(init) => write!(f, "declared {init}"),
            NoInit => write!(f, "uninit"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticInit {
    Int(i32),
    Long(i64),
}
impl StaticInit {
    pub fn is_zero(self) -> bool {
        matches!(self, StaticInit::Int(0) | StaticInit::Long(0))
    }
}
impl std::fmt::Display for StaticInit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StaticInit::Int(i) => write!(f, "long   {i}"),
            StaticInit::Long(i) => write!(f, "quad   {i}"),
        }
    }
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
    pub init: Option<TypedExpr>,
    pub sc: StorageClass,
    pub var_type: Type,
}
impl VarDecl {
    fn type_check_file(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        let mut init = match self.init.clone() {
            Some(TypedExpr { expr: Expr::Const(cnst), .. }) => {
                Initial(cnst.into_static_init(self.var_type.clone()))
            }
            None if self.sc == StorageClass::Extern => NoInit,
            None => Tentative,
            _ => anyhow::bail!("non-constant initializer"),
        };

        let mut global = self.sc != StorageClass::Static;

        match symbols.get(&self.name) {
            None => {}
            Some(TypeCtx { type_: Type::Func { .. }, .. }) => {
                // not a constant
                anyhow::bail!("function redclared as variable");
            }
            Some(TypeCtx { type_, .. }) if *type_ != self.var_type => {
                // not a constant
                anyhow::bail!("conflicting type declarations");
            }
            Some(TypeCtx { attr: Local | Func { .. }, .. }) => {
                unreachable!()
            }
            Some(TypeCtx { attr: Static { init: o_i, global: o_g }, .. }) => {
                match self.sc {
                    StorageClass::Extern => global = *o_g,
                    _ if *o_g != global => anyhow::bail!("conflicting variable linkage"),
                    _ => {}
                }

                match (init, *o_i) {
                    (Initial(_), Initial(_)) => anyhow::bail!("conflicting file scope definitions"),
                    (Initial(i), _) | (_, Initial(i)) => init = Initial(i),
                    (Tentative, _) | (_, Tentative) => init = Tentative,
                    (NoInit, NoInit) => {}
                };
            }
        }

        symbols.insert(
            self.name.clone(),
            TypeCtx { type_: self.var_type.clone(), attr: Static { init, global } },
        );

        Ok(self)
    }
    fn type_check_block(mut self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        match self.sc {
            StorageClass::Extern if self.init.is_some() => {
                anyhow::bail!("init on local extern variable");
            }
            StorageClass::Extern => match symbols.entry(self.name.clone()) {
                Entry::Occupied(occ) if occ.get().type_ != self.var_type.clone() => {
                    anyhow::bail!("conflicting declaration of variable");
                }
                Entry::Vacant(vac) => {
                    vac.insert(TypeCtx {
                        type_: self.var_type.clone(),
                        attr: Static { init: NoInit, global: true },
                    });
                }
                Entry::Occupied(_) => {}
            },
            StorageClass::Static => {
                let init_value = match self.init.take() {
                    Some(TypedExpr { expr: Expr::Const(cnst), .. }) => {
                        Initial(cnst.into_static_init(self.var_type.clone()))
                    }
                    None => Initial(StaticInit::Int(0)),
                    _ => anyhow::bail!("non-constant initializer on local static variable"),
                };

                symbols.insert(
                    self.name.clone(),
                    TypeCtx {
                        type_: self.var_type.clone(),
                        attr: Static { init: init_value, global: false },
                    },
                );
            }
            StorageClass::None => {
                symbols.insert(
                    self.name.clone(),
                    TypeCtx { type_: self.var_type.clone(), attr: Local },
                );

                self.init = match self.init {
                    Some(v) => Some(v.type_check(symbols)?),
                    None => None,
                };
            }
        }

        Ok(self)
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

        Ok(VarDecl { name, init, sc, var_type: self.var_type })
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

        let Type::Func { ret: ret_type, params: arg_types } = self.fun_type.clone() else {
            unreachable!()
        };

        match symbols.get(&self.name) {
            None => {}
            Some(TypeCtx { type_: Type::Func { .. }, attr: Func { defined, .. } })
                if *defined && has_body =>
            {
                anyhow::bail!("multiple function definitions")
            }

            Some(TypeCtx { type_: Type::Func { .. }, attr: Func { global, .. } })
                if *global && self.sc == StorageClass::Static =>
            {
                anyhow::bail!("Static function declaration follows non-static")
            }

            Some(TypeCtx {
                type_: type_ @ Type::Func { params, .. },
                attr: Func { defined, global: old_global },
            }) if params.len() == self.params.len() && *type_ == self.fun_type => {
                already_defined = *defined;
                global = *old_global;
            }
            _ => anyhow::bail!("incompatible function declarations"),
        }

        symbols.insert(
            self.name.clone(),
            TypeCtx {
                type_: self.fun_type.clone(),
                attr: Func { defined: already_defined || has_body, global },
            },
        );

        let body = if let Some(body) = self.body {
            for (param, ty) in self.params.clone().iter().zip(arg_types) {
                symbols.insert(param.clone(), TypeCtx { type_: ty, attr: Local });
            }

            Some(body.type_check(symbols, *ret_type)?)
        } else {
            None
        };

        Ok(Self {
            name: self.name,
            params: self.params,
            body,
            sc: if global { StorageClass::Extern } else { StorageClass::Static },
            fun_type: self.fun_type,
        })
    }

    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> anyhow::Result<Self> {
        if map
            .insert(self.name.clone(), IdCtx::new(self.name.clone(), true, true))
            .is_some_and(|id| id.in_current_scope && !id.has_linkage)
        {
            anyhow::bail!("duplicate declaration. listing 9-19");
        };

        let Type::Func { params: ref params_type, .. } = self.fun_type else { unreachable!() };

        let mut inner_map = map
            .iter()
            .map(|(k, idctx)| (k.clone(), IdCtx { in_current_scope: false, ..idctx.clone() }))
            .collect();

        let mut params = Vec::with_capacity(self.params.len());
        for (name, ty) in self.params.iter().zip(params_type) {
            let VarDecl { name, .. } = (VarDecl {
                name: name.clone(),
                init: None,
                sc: StorageClass::None,
                var_type: ty.clone(),
            })
            .resolve_identifiers(&mut inner_map)?;

            params.push(name);
        }

        let body = match self.body {
            Some(body) => Some(body.resolve_identifiers(&mut inner_map)?),
            None => None,
        };

        Ok(Self { name: self.name, params, body, sc: self.sc, fun_type: self.fun_type })
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

        Ok(Self {
            name: self.name,
            body,
            params: self.params,
            sc: self.sc,
            fun_type: self.fun_type,
        })
    }

    fn resolve_loop_labels(self) -> anyhow::Result<Self> {
        let body = match self.body {
            Some(body) => Some(body.resolve_loop_labels(LoopKind::None)?),
            None => None,
        };

        Ok(Self {
            name: self.name,
            body,
            params: self.params,
            sc: self.sc,
            fun_type: self.fun_type,
        })
    }

    fn resolve_switch_statements(self) -> anyhow::Result<Self> {
        let switch_type = Type::Int; // doesn't actually matter
        let body = match self.body {
            Some(body) => Some(body.resolve_switch_statements(None, &switch_type)?),
            None => None,
        };

        Ok(Self {
            name: self.name,
            body,
            params: self.params,
            sc: self.sc,
            fun_type: self.fun_type,
        })
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

    fn resolve_switch_statements(
        self,
        switch_ctx: SwitchCtx,
        switch_type: &Type,
    ) -> anyhow::Result<Self> {
        Ok(match self {
            Self::S(stmt) => Self::S(stmt.resolve_switch_statements(switch_ctx, switch_type)?),
            Self::D(_) => self,
        })
    }

    fn type_check(
        self,
        symbols: &mut Namespace<TypeCtx>,
        enclosing_func_ret: Type,
    ) -> anyhow::Result<Self> {
        Ok(match self {
            Self::S(stmt) => Self::S(stmt.type_check(symbols, enclosing_func_ret)?),
            Self::D(decl) => Self::D(decl.type_check(Scope::Block, symbols)?),
        })
    }
}

type SwitchCtx<'s> = Option<&'s mut FxHashSet<Option<Const>>>;

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

    fn resolve_switch_statements(
        self,
        switch_ctx: SwitchCtx,
        switch_type: &Type,
    ) -> anyhow::Result<Self> {
        let mut acc = Vec::with_capacity(self.0.len());

        if let Some(switch_ctx) = switch_ctx {
            for bi in self.0 {
                let bi = bi.resolve_switch_statements(Some(switch_ctx), switch_type)?;
                acc.push(bi);
            }
        } else {
            for bi in self.0 {
                let bi = bi.resolve_switch_statements(None, switch_type)?;
                acc.push(bi);
            }
        }

        Ok(Self(acc))
    }

    fn type_check(
        self,
        symbols: &mut Namespace<TypeCtx>,
        enclosing_func_ret: Type,
    ) -> anyhow::Result<Self> {
        let mut acc = Vec::with_capacity(self.0.len());

        for bi in self.0 {
            let bi = bi.type_check(symbols, enclosing_func_ret.clone())?;
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
    Return(TypedExpr),
    Expression(TypedExpr),
    If {
        cond: TypedExpr,
        then: Box<Stmt>,
        else_: Option<Box<Stmt>>,
    },

    Compound(Block),

    Break(Option<Ecow>),
    Continue(Option<Ecow>),
    While {
        cond: TypedExpr,
        body: Box<Stmt>,
        label: Option<Ecow>,
    },
    DoWhile {
        body: Box<Stmt>,
        cond: TypedExpr,
        label: Option<Ecow>,
    },
    For {
        init: Either<VarDecl, Option<TypedExpr>>,
        cond: Option<TypedExpr>,
        post: Option<TypedExpr>,

        body: Box<Stmt>,
        label: Option<Ecow>,
    },

    // extra credit
    GoTo(Ecow),
    Label(Ecow, Box<Stmt>),

    Switch {
        ctrl: TypedExpr,
        body: Box<Stmt>,
        label: Option<Ecow>,
        cases: Vec<Option<Const>>,
    },
    Case {
        cnst: TypedExpr,
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

    fn resolve_switch_statements(
        self,
        switch_ctx: SwitchCtx,
        switch_type: &Type,
    ) -> anyhow::Result<Self> {
        Ok(match self {
            Self::If { cond, then, else_ } => {
                let mut switch_ctx = switch_ctx;

                let then = Box::new(
                    then.resolve_switch_statements(switch_ctx.as_deref_mut(), switch_type)?,
                );
                let else_ = match else_ {
                    Some(e) => {
                        Some(Box::new(e.resolve_switch_statements(switch_ctx, switch_type)?))
                    }
                    None => None,
                };

                Self::If { cond, then, else_ }
            }
            Self::Compound(block) => {
                Self::Compound(block.resolve_switch_statements(switch_ctx, switch_type)?)
            }
            Self::While { cond, body, label } => {
                let body = Box::new(body.resolve_switch_statements(switch_ctx, switch_type)?);
                Self::While { cond, body, label }
            }
            Self::DoWhile { body, cond, label } => {
                let body = Box::new(body.resolve_switch_statements(switch_ctx, switch_type)?);
                Self::DoWhile { cond, body, label }
            }
            Self::For { init, cond, post, body, label } => {
                let body = Box::new(body.resolve_switch_statements(switch_ctx, switch_type)?);
                Self::For { init, cond, post, body, label }
            }
            Self::Label(label, stmt) => {
                let stmt = Box::new(stmt.resolve_switch_statements(switch_ctx, switch_type)?);
                Self::Label(label, stmt)
            }

            Self::Switch { ctrl, body, label, .. } => {
                let switch_type = ctrl
                    .clone()
                    .ret
                    .ok_or(anyhow::anyhow!("switch type must be kbown at this point"))?;

                let mut switchctx = HashSet::default();
                let body =
                    Box::new(body.resolve_switch_statements(Some(&mut switchctx), &switch_type)?);

                let cases = switchctx.into_iter().collect::<Vec<_>>();

                Self::Switch { ctrl, body, label, cases }
            }
            Self::Case { cnst, body, label } => {
                let switch_ctx = switch_ctx.ok_or(anyhow::anyhow!("case outside of a switch"))?;

                let Expr::Const(value) = cnst.expr else {
                    anyhow::bail!("case value not a constant");
                };

                let value = value.cast_const(switch_type);

                if !switch_ctx.insert(Some(value)) {
                    anyhow::bail!("case value already exists");
                };

                let body = Box::new(body.resolve_switch_statements(Some(switch_ctx), switch_type)?);
                Self::Case { cnst, body, label }
            }
            Self::Default { body, label } => {
                let switch_ctx =
                    switch_ctx.ok_or(anyhow::anyhow!("default outside of a switch"))?;

                if !switch_ctx.insert(None) {
                    anyhow::bail!("cdefault ase already exists");
                };

                let body = Box::new(body.resolve_switch_statements(Some(switch_ctx), switch_type)?);
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

    fn type_check(
        self,
        symbols: &mut Namespace<TypeCtx>,
        enclosing_func_ret: Type,
    ) -> anyhow::Result<Self> {
        Ok(match self {
            Self::Return(expr) => {
                let expr = expr.type_check(symbols)?.cast_to_type(enclosing_func_ret);

                Self::Return(expr)
            }
            Self::Expression(expr) => Self::Expression(expr.type_check(symbols)?),
            Self::If { cond, then, else_ } => Self::If {
                cond: cond.type_check(symbols)?,
                then: Box::new(then.type_check(symbols, enclosing_func_ret.clone())?),
                else_: match else_ {
                    Some(s) => Some(Box::new(s.type_check(symbols, enclosing_func_ret)?)),
                    None => None,
                },
            },
            Self::Compound(block) => Self::Compound(block.type_check(symbols, enclosing_func_ret)?),
            Self::While { cond, body, label } => Self::While {
                cond: cond.type_check(symbols)?,
                body: Box::new(body.type_check(symbols, enclosing_func_ret)?),
                label,
            },
            Self::DoWhile { body, cond, label } => Self::DoWhile {
                body: Box::new(body.type_check(symbols, enclosing_func_ret)?),
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
                body: Box::new(body.type_check(symbols, enclosing_func_ret)?),
                label,
            },
            Self::Label(label, stmt) => {
                Self::Label(label, Box::new(stmt.type_check(symbols, enclosing_func_ret)?))
            }
            Self::Switch { ctrl, body, label, cases } => Self::Switch {
                ctrl: ctrl.type_check(symbols)?,
                body: Box::new(body.type_check(symbols, enclosing_func_ret)?),
                label,
                cases,
            },
            Self::Case { cnst, body, label } => Self::Case {
                cnst: cnst.type_check(symbols)?,
                body: Box::new(body.type_check(symbols, enclosing_func_ret)?),
                label,
            },
            Self::Default { body, label } => Self::Default {
                body: Box::new(body.type_check(symbols, enclosing_func_ret)?),
                label,
            },
            Self::GoTo(_) | Self::Break(_) | Self::Continue(_) | Self::Null => self,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedExpr {
    pub expr: Expr,
    pub ret: Option<Type>,
}
impl TypedExpr {
    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> anyhow::Result<Self> {
        Ok(Self { expr: self.expr.resolve_identifiers(map)?, ret: self.ret })
    }

    fn type_check(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        self.expr.type_check(symbols)
    }

    fn cast_to_type(self, target: Type) -> Self {
        if self.ret == Some(target.clone()) {
            return self;
        }
        Expr::Cast { target: target.clone(), inner: Box::new(self) }.typed(target)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Const(Const),
    Var(Ecow),
    Cast { target: Type, inner: Box<TypedExpr> },
    Unary(UnaryOp, Box<TypedExpr>),
    Binary { op: BinaryOp, lhs: Box<TypedExpr>, rhs: Box<TypedExpr> },
    CompoundAssignment { op: BinaryOp, lhs: Box<TypedExpr>, rhs: Box<TypedExpr> },
    Assignemnt(Box<TypedExpr>, Box<TypedExpr>),
    Conditional { cond: Box<TypedExpr>, then: Box<TypedExpr>, else_: Box<TypedExpr> },
    FuncCall { name: Ecow, args: Vec<TypedExpr> },
}
impl Expr {
    pub fn dummy_typed(self) -> TypedExpr {
        TypedExpr { expr: self, ret: None }
    }
    pub fn typed(self, ty: Type) -> TypedExpr {
        TypedExpr { expr: self, ret: Some(ty) }
    }

    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> anyhow::Result<Self> {
        match self {
            Self::Assignemnt(left, right) if matches!(left.expr, Expr::Var(_)) => {
                Ok(Self::Assignemnt(
                    Box::new(left.resolve_identifiers(map)?),
                    Box::new(right.resolve_identifiers(map)?),
                ))
            }
            Self::Assignemnt(_, _) => anyhow::bail!("left value isn't a variable"),

            Self::Unary(op, expr)
                if matches!(
                    op,
                    UnaryOp::IncPre | UnaryOp::IncPost | UnaryOp::DecPre | UnaryOp::DecPost
                ) && !matches!(expr.expr, Expr::Var(_)) =>
            {
                anyhow::bail!("left value to increment/decrement isn't a variable");
            }
            Self::Unary(op, expr) => Ok(Self::Unary(op, Box::new(expr.resolve_identifiers(map)?))),

            Self::Binary { op, lhs, rhs } => Ok(Self::Binary {
                op,
                lhs: Box::new(lhs.resolve_identifiers(map)?),
                rhs: Box::new(rhs.resolve_identifiers(map)?),
            }),

            Self::CompoundAssignment { lhs, .. } if !matches!(lhs.expr, Expr::Var(_)) => {
                anyhow::bail!("left value to compound assignment isn't a variable");
            }
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
            Self::Cast { target: to, inner: from } => {
                let from = Box::new(from.resolve_identifiers(map)?);
                Ok(Self::Cast { target: to, inner: from })
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn type_check(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<TypedExpr> {
        match self {
            Self::FuncCall { name, args } => {
                let (types, ret) = match &symbols
                    .get(&name)
                    .ok_or(anyhow::anyhow!("function does not exist in symbol map"))?
                    .type_
                {
                    Type::Func { params, .. } if params.len() != args.len() => {
                        anyhow::bail!("function called with wrong number of arguments");
                    }
                    Type::Func { params, ret } => (params.clone(), ret.deref().clone()),

                    _ => anyhow::bail!("Variable used as function name"),
                };

                let mut acc = Vec::with_capacity(types.len());
                for (arg, ty) in args.iter().cloned().zip(types) {
                    let arg = arg.type_check(symbols)?;
                    acc.push(arg.cast_to_type(ty));
                }

                Ok(Self::FuncCall { name, args: acc }.typed(ret))
            }
            Self::Var(name) => match symbols.get(&name).map(|c| c.type_.clone()) {
                Some(Type::Func { .. }) => {
                    anyhow::bail!("function used as variable");
                }
                None => {
                    anyhow::bail!("variable name is not declared");
                }
                Some(ty) => Ok(Self::Var(name).typed(ty)),
            },

            // --
            Self::Unary(op, expr) => {
                let expr = Box::new(expr.type_check(symbols)?);

                let ty = match op {
                    UnaryOp::Not => Type::Int,
                    _ => expr.clone().ret.expect("unary type should be known"),
                };

                Ok(Self::Unary(op, expr).typed(ty))
            }

            Self::Binary { op, lhs, rhs } => {
                let lhs = Box::new(lhs.type_check(symbols)?);
                let rhs = Box::new(rhs.type_check(symbols)?);

                match op {
                    BinaryOp::And | BinaryOp::Or => {
                        return Ok(Self::Binary { op, lhs, rhs }.typed(Type::Int));
                    }
                    _ => {}
                }

                let common = lhs
                    .clone()
                    .ret
                    .and_then(|lht| rhs.clone().ret.map(|rht| lht.get_common_type(rht)))
                    .expect("binary operand type should be known at this point");

                let lhs = Box::new(lhs.cast_to_type(common.clone()));
                let rhs = Box::new(rhs.cast_to_type(common.clone()));

                let ret = Self::Binary { op, lhs, rhs };

                Ok(match op {
                    BinaryOp::Add
                    | BinaryOp::Subtract
                    | BinaryOp::Multiply
                    | BinaryOp::Divide
                    | BinaryOp::Reminder => ret.typed(common),

                    _ => ret.typed(Type::Int),
                })
            }

            Self::Assignemnt(lhs, rhs) => {
                let lhs = Box::new(lhs.type_check(symbols)?);
                let rhs = rhs.type_check(symbols)?;

                let left_type =
                    lhs.clone().ret.expect("assignee type should be known at this point");
                let rhs = Box::new(rhs.cast_to_type(left_type.clone()));

                Ok(Self::Assignemnt(lhs, rhs).typed(left_type))
            }

            Self::CompoundAssignment { op, lhs, rhs } => {
                let lhs = Box::new(lhs.type_check(symbols)?);
                let rhs = Box::new(rhs.type_check(symbols)?);

                // same logic as binary currently with slight changes
                // most likely wrong . todo
                // shut up rustc for now
                // according to Claude, the caluclation is done with `long`
                // but type of lhs does not change so it is caast back to int.

                match op {
                    BinaryOp::And | BinaryOp::Or => {
                        return Ok(Self::CompoundAssignment { op, lhs, rhs }.typed(Type::Int));
                    }
                    _ => {}
                }

                let left_type =
                    lhs.clone().ret.expect("assignee type should be known at this point");

                let common = lhs
                    .clone()
                    .ret
                    .and_then(|lht| rhs.clone().ret.map(|rht| lht.get_common_type(rht)))
                    .expect("compound assignment operand type should be known at this point");

                let lhs = Box::new(lhs.cast_to_type(common.clone()));
                let rhs = Box::new(rhs.cast_to_type(common.clone()));

                let ret = Self::CompoundAssignment { op, lhs, rhs };

                Ok(match op {
                    BinaryOp::Add
                    | BinaryOp::Subtract
                    | BinaryOp::Multiply
                    | BinaryOp::Divide
                    | BinaryOp::Reminder => ret.typed(left_type),

                    _ => ret.typed(Type::Int),
                })
            }
            Self::Const(c) => match c {
                Const::Int(_) => Ok(self.typed(Type::Int)),
                Const::Long(_) => Ok(self.typed(Type::Long)),
            },
            Self::Conditional { cond, then, else_ } => {
                let cond = Box::new(cond.type_check(symbols)?);
                let then = then.type_check(symbols)?;
                let else_ = else_.type_check(symbols)?;

                let common = then
                    .clone()
                    .ret
                    .and_then(|lht| else_.clone().ret.map(|rht| lht.get_common_type(rht)))
                    .expect("ternary operand type should be known at this point");

                let then = Box::new(then.cast_to_type(common.clone()));
                let else_ = Box::new(else_.cast_to_type(common.clone()));

                Ok(Self::Conditional { cond, then, else_ }.typed(common))
            }

            Self::Cast { target, inner } => {
                let inner = Box::new(inner.type_check(symbols)?);
                Ok(Self::Cast { target: target.clone(), inner }.typed(target))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Const {
    Int(i32),
    Long(i64),
}

impl std::fmt::Display for Const {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // this is used for assembly generation for some reason.
        match self {
            Const::Int(i) => write!(f, "{i}"),
            Const::Long(i) => write!(f, "{i}"),
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
impl Const {
    fn into_static_init(self, target_type: Type) -> StaticInit {
        match (self, target_type) {
            (_, Type::Func { .. }) => unreachable!(),

            (Const::Int(i), Type::Int) => StaticInit::Int(i),
            (Const::Int(i), Type::Long) => StaticInit::Long(i64::from(i)),
            (Const::Long(i), Type::Int) => StaticInit::Int(i as i32),
            (Const::Long(i), Type::Long) => StaticInit::Long(i),
        }
    }
    fn cast_const(self, target_type: &Type) -> Self {
        match (self, target_type) {
            (_, Type::Func { .. }) => unreachable!(),

            (Const::Int(i), Type::Int) => Self::Int(i),
            (Const::Int(i), Type::Long) => Self::Long(i64::from(i)),
            (Const::Long(i), Type::Int) => Self::Int(i as i32),
            (Const::Long(i), Type::Long) => Self::Long(i),
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

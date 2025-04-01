use crate::ir::{self, GEN};
use ecow::{EcoString as Identifier, eco_format};
use either::Either::{self, Left, Right};
use indexmap::{IndexMap, IndexSet, map::Entry};
use std::{
    fmt::{Display, Formatter, Write},
    hash::Hash,
    ops::Deref,
    sync::atomic::Ordering::Relaxed,
};

// IndexMap retains insertion order so it prints consistently
pub type Namespace<T> = IndexMap<Identifier, T, rustc_hash::FxBuildHasher>;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct IdCtx {
    name: Identifier,
    in_current_scope: bool,
    has_linkage: bool,
}
impl IdCtx {
    fn new(name: Identifier, in_current_scope: bool, has_linkage: bool) -> Self {
        Self { name, in_current_scope, has_linkage }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeCtx {
    pub type_: Type,
    pub attr: Attributes,
}
impl Display for TypeCtx {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let attr = match &self.attr {
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
impl Display for Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PROGRAM")?;
        for decl in &self.decls {
            writeln!(f, "{decl:1}")?;
        }

        Ok(())
    }
}
impl Program {
    pub fn new(decls: Vec<Decl>) -> Self {
        Self { decls, symbols: Namespace::default() }
    }

    pub fn semantic_analysis(mut self) -> anyhow::Result<Self> {
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

        Ok(self)
    }

    pub fn compile(mut self) -> ir::Program {
        let mut buf = Vec::new();
        let mut symbols = std::mem::take(&mut self.symbols);

        self.to_ir(&mut buf, &mut symbols)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Long,
    UInt,
    ULong,
    Double,
    Func { params: Vec<Type>, ret: Box<Type> },
    Pointer { to: Box<Type> },
    Array { element: Box<Type>, size: usize },
}
impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int => f.pad("int"),
            Self::Long => f.pad("long"),
            Self::UInt => f.pad("uint"),
            Self::ULong => f.pad("ulong"),
            Self::Double => f.pad("double"),
            Self::Func { params, ret } => {
                let mut buf = Identifier::new();
                write!(buf, "func (")?;
                for (idx, param) in params.iter().enumerate() {
                    write!(buf, "{}{param}", if idx != 0 { ", " } else { "" })?;
                }
                write!(buf, ") -> {ret}")?;

                f.pad(&buf)
            }
            Self::Pointer { to } => {
                let buf = eco_format!("ptr->({to})");
                f.pad(&buf)
            }
            Self::Array { element, size } => {
                let buf = eco_format!("[{element}; {size}]");
                f.pad(&buf)
            }
        }
    }
}

impl Type {
    pub fn size(&self) -> usize {
        match self {
            Self::Int | Self::UInt => 4,
            Self::Long | Self::ULong | Self::Pointer { .. } | Type::Double => 8,
            Self::Func { .. } => unreachable!(
                "function types don't have size. why is function in the same type anyway ?"
            ),

            Self::Array { element, size } => size * element.size(),
        }
    }
    pub fn signed(&self) -> bool {
        match self {
            Self::Int | Self::Long => true,
            Self::UInt | Self::ULong | Self::Pointer { .. } => false,
            Self::Func { .. } | Self::Array { .. } => unreachable!(
                "function and Array types don't have sign. why is function in the same type anyway ?"
            ),
            Self::Double => unreachable!("doubled signedness unused for IR"),
        }
    }
    fn get_common_type(self, other: Self) -> Self {
        // why is function in the same type anyway ?
        if self == other {
            self
        } else if self == Self::Double || other == Self::Double {
            Self::Double
        } else if self.size() == other.size() {
            if self.signed() { other } else { self }
        } else if self.size() > other.size() {
            self
        } else {
            other
        }
    }
    pub fn zeroed_static(&self) -> StaticInit {
        match self {
            Self::Int => StaticInit::Int(0),
            Self::Long => StaticInit::Long(0),
            Self::UInt => StaticInit::UInt(0),
            Self::ULong | Self::Pointer { .. } => StaticInit::ULong(0),
            Self::Func { .. } => unreachable!("function static value not a thing"),

            Self::Double => StaticInit::Double(0.0),
            Self::Array { element, size } => StaticInit::Zero(size * element.size()),
        }
    }
    pub fn zeroed_init(&self) -> Initializer {
        macro_rules! zi {
            ($ty:tt) => {
                Initializer::Single(TypedExpr {
                    expr: Expr::Const(Const::$ty(0 as _)),
                    type_: Some(Self::$ty),
                })
            };
        }
        match self {
            Self::Int => zi!(Int),
            Self::Long => zi!(Long),
            Self::UInt => zi!(UInt),
            Self::ULong | Self::Pointer { .. } => zi!(ULong),
            Self::Func { .. } => unreachable!("function zero values are not a thing"),

            Self::Double => zi!(Double),
            Self::Array { element, size } => {
                Initializer::Compound(vec![element.zeroed_init(); *size])
            }
        }
    }
    pub fn is_intish(&self) -> bool {
        matches!(self, Self::Int | Self::Long | Self::UInt | Self::ULong)
    }
    fn is_arithmatic(&self) -> bool {
        matches!(self, Self::Int | Self::Long | Self::UInt | Self::ULong | Self::Double)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Attributes {
    Func { defined: bool, global: bool },
    Static { init: InitValue, global: bool },
    Local,
}
pub use Attributes::*;

#[derive(Debug, Clone, PartialEq)]
pub enum InitValue {
    Tentative,
    Initial(Vec<StaticInit>),
    NoInit,
}
pub use InitValue::*;
impl Display for InitValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Tentative => write!(f, "tentative"),
            Initial(inits) => {
                write!(f, "declared {{ ")?;
                for (idx, init) in inits.iter().enumerate() {
                    if idx > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{init}")?;
                }
                write!(f, " }}")
            }
            NoInit => write!(f, "uninit"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StaticInit {
    Int(i32),
    Long(i64),
    UInt(u32),
    ULong(u64),
    Double(f64),
    Zero(usize),
}
impl StaticInit {
    pub fn is_zero(self) -> bool {
        matches!(self, |Self::Int(0)| Self::Long(0) | Self::UInt(0) | Self::ULong(0))
    }
}
impl Display for StaticInit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            // this is used for assembly
            Self::Int(i) => write!(f, "long   {i}"),
            Self::Long(i) => write!(f, "quad   {i}"),
            Self::UInt(i) => write!(f, "long   {i}"),
            Self::ULong(i) => write!(f, "quad   {i}"),

            Self::Double(i) => write!(f, "quad   {}", i.to_bits()),
            Self::Zero(i) => write!(f, "todo   {i}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    Static,
    Extern,
    None,
}
impl Display for StorageClass {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static => f.pad("static"),
            Self::Extern => f.pad("extern"),
            Self::None => Ok(()),
        }
    }
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
impl Display for Decl {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let indent = f.width().unwrap_or_default();
        let pad = Identifier::from("\t").repeat(indent);

        match self {
            Self::Func(func) => write!(f, "{pad}{func:indent$}"),
            Self::Var(var) => write!(f, "{pad}{var:indent$}"),
        }
    }
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
            Self::Func(func_decl) => Self::Func(func_decl.resolve_switch_statements()?),
            Self::Var(_) => self,
        })
    }

    fn resolve_goto_labels(self) -> anyhow::Result<Self> {
        Ok(match self {
            Self::Func(func_decl) => Self::Func(func_decl.resolve_goto_labels()?),
            Self::Var(_) => self,
        })
    }

    fn resolve_loop_labels(self) -> anyhow::Result<Self> {
        Ok(match self {
            Self::Func(func_decl) => Self::Func(func_decl.resolve_loop_labels()?),
            Self::Var(_) => self,
        })
    }
}

#[derive(Debug, Clone)]
pub enum Initializer {
    Single(TypedExpr),
    Compound(Vec<Initializer>),
}
impl Display for Initializer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(expr) => write!(f, "{}", expr),
            Self::Compound(inits) => {
                for (idx, init) in inits.iter().enumerate() {
                    write!(f, "{}{init}", if idx != 0 { ", " } else { "{ " })?;
                }
                write!(f, " }}")
            }
        }
    }
}
impl Initializer {
    pub fn flatten_exprs (self) -> Vec<TypedExpr> {
        match self {
            Initializer::Single(e) => vec![e],
            Initializer::Compound(inits) => inits.into_iter().flat_map(|i| i.flatten_exprs()).collect(),
        }
    }
    fn type_check(self, symbols: &mut Namespace<TypeCtx>, target: Type) -> anyhow::Result<Self> {
        Ok(match (&target, self) {
            (_, Self::Single(expr)) => {
                let expr = expr.type_check_and_convert(symbols)?.cast_by_assignment(target)?;
                Self::Single(expr)
            }
            (Type::Array { element, size }, Self::Compound(inits)) => {
                anyhow::ensure!(inits.len() <= *size, "wrong number of values in initializer");

                let mut inits = inits
                    .into_iter()
                    .map(|i| i.type_check(symbols, *element.clone()))
                    .collect::<Result<Vec<_>, _>>()?;

                while inits.len() < *size {
                    inits.push(element.zeroed_init());
                }

                Self::Compound(inits)
            }
            _ => anyhow::bail!("cannot initialize a scalar with a compound init"),
        })
    }

    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> anyhow::Result<Self> {
        Ok(match self {
            Self::Single(expr) => Self::Single(expr.resolve_identifiers(map)?),
            Self::Compound(inits) => {
                let inits = inits
                    .into_iter()
                    .map(|i| i.resolve_identifiers(map))
                    .collect::<Result<_, _>>()?;
                Self::Compound(inits)
            }
        })
    }

    pub fn to_init_value(&self, var_type: Type) -> anyhow::Result<Vec<StaticInit>> {
        let init_value = match self {
            Self::Single(TypedExpr { expr: Expr::Const(cnst), .. }) => {
                vec![cnst.into_static_init(&var_type)]
            }
            Self::Single(TypedExpr { expr: Expr::Cast { inner, .. }, .. }) => {
                return Self::Single(*inner.clone()).to_init_value(var_type);
            }
            Self::Compound(inits) if matches!(var_type, Type::Array { .. }) => {
                let Type::Array { element, size } = var_type.clone() else { unreachable!() };

                let mut inits: Vec<_> = inits
                    .iter()
                    .map(|i| i.to_init_value(*element.clone()))
                    .collect::<Result<_, _>>()?;

                if inits.len() < size {
                    inits.push(vec![StaticInit::Zero(size * var_type.size())]);
                }

                inits.into_iter().flatten().collect()
            }
            _ => anyhow::bail!("non-constant initializer"),
        };

        Ok(init_value)
    }
}

#[derive(Debug, Clone)]
pub struct VarDecl {
    pub name: Identifier,
    pub init: Option<Initializer>,
    pub sc: StorageClass,
    pub var_type: Type,
}
impl Display for VarDecl {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ln = eco_format!("{} {} {}", self.sc, self.var_type, self.name);
        let ln = ln.trim();
        write!(f, "{ln}",)?;

        if let Some(init) = &self.init {
            write!(f, " = {init}")?;
        }

        Ok(())
    }
}
impl VarDecl {
    fn type_check_file(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        let mut init = match self.init {
            Some(ref inits) => Initial(inits.to_init_value(self.var_type.clone())?),
            None if self.sc == StorageClass::Extern => NoInit,
            None => Tentative,
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

                match (init.clone(), o_i.clone()) {
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
                    Some(init) => Initial(
                        init.type_check(symbols, self.var_type.clone())?
                            .to_init_value(self.var_type.clone())?,
                    ),
                    None => Initial(vec![StaticInit::Int(0)]),
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
                    Some(v) => Some(v.type_check(symbols, self.var_type.clone())?),
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

        Ok(Self { name, init, sc, var_type: self.var_type })
    }
}

#[derive(Debug, Clone)]
pub struct FuncDecl {
    pub name: Identifier,
    pub params: Vec<Identifier>,
    pub body: Option<Block>,
    pub sc: StorageClass,
    pub fun_type: Type,
}
impl Display for FuncDecl {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let indent = f.width().unwrap_or_default();
        let child = indent + 1;

        let Type::Func { ret, params } = &self.fun_type else { unreachable!() };

        let ln = eco_format!("{} {} {} (", self.sc, ret, self.name);
        let ln = ln.trim();

        write!(f, "{ln}")?;
        for (idx, (ty, name)) in params.iter().zip(&self.params).enumerate() {
            write!(f, "{}{ty} {name}", if idx != 0 { ", " } else { "" })?;
        }
        write!(f, ")")?;

        if let Some(body) = &self.body {
            writeln!(f)?;
            for item in &body.0 {
                writeln!(f, "{item:child$}")?;
            }
        }

        Ok(())
    }
}
impl FuncDecl {
    fn type_check(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        let has_body = self.body.is_some();
        let mut already_defined = false;
        let mut global = self.sc != StorageClass::Static;

        let Type::Func { ret: ret_type, params: arg_types } = self.fun_type.clone() else {
            unreachable!()
        };

        anyhow::ensure!(
            !matches!(*ret_type, Type::Array { .. }),
            "a function cannot return an array"
        );
        let arg_types = arg_types.into_iter().map(|t| match t {
            Type::Array { element, .. } => Type::Pointer { to: element },
            _ => t,
        });
        let fun_type = Type::Func { params: arg_types.clone().collect(), ret: ret_type.clone() };

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
            }) if params.len() == self.params.len() && *type_ == fun_type => {
                already_defined = *defined;
                global = *old_global;
            }
            _ => anyhow::bail!("incompatible function declarations"),
        }

        symbols.insert(
            self.name.clone(),
            TypeCtx {
                type_: fun_type.clone(),
                attr: Func { defined: already_defined || has_body, global },
            },
        );

        let body = if let Some(body) = self.body {
            for (param, ty) in self.params.iter().zip(arg_types) {
                symbols.insert(param.clone(), TypeCtx { type_: ty, attr: Local });
            }

            Some(body.type_check(symbols, &ret_type)?)
        } else {
            None
        };

        Ok(Self {
            name: self.name,
            params: self.params,
            body,
            sc: if global { StorageClass::Extern } else { StorageClass::Static },
            fun_type,
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
        let mut label_map = IndexMap::default();
        let body = match self.body {
            Some(body) => Some(body.resolve_goto_labels(&mut label_map, &self.name)?),
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
            Some(body) => Some(body.resolve_loop_labels(&LoopKind::None)?),
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
impl Display for BlockItem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let indent = f.width().unwrap_or_default();
        match self {
            Self::S(stmt) => write!(f, "{stmt:indent$}"),
            Self::D(decl) => write!(f, "{decl:indent$}"),
        }
    }
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
        func_name: Identifier,
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

type SwitchCtx<'s> = Option<&'s mut IndexSet<Option<Const>, rustc_hash::FxBuildHasher>>;

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
        func_name: &Identifier,
    ) -> anyhow::Result<Self> {
        let mut acc = Vec::with_capacity(self.0.len());
        for bi in self.0 {
            let bi = bi.resolve_goto_labels(labels, func_name.clone())?;
            acc.push(bi);
        }

        Ok(Self(acc))
    }

    fn resolve_loop_labels(self, current_label: &LoopKind) -> anyhow::Result<Self> {
        let mut acc = Vec::with_capacity(self.0.len());

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
        enclosing_func_ret: &Type,
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
    Loop(Identifier),
    Switch(Identifier),
    SwitchInLoop { loop_: Identifier, switch: Identifier },
    LoopInSwitch { loop_: Identifier, switch: Identifier },
    None,
}
impl LoopKind {
    fn break_label(&self) -> Option<Identifier> {
        match self {
            Self::Loop(n)
            | Self::Switch(n)
            | Self::SwitchInLoop { switch: n, .. }
            | Self::LoopInSwitch { loop_: n, .. } => Some(n.clone()),
            Self::None => None,
        }
    }
    fn loop_label(&self) -> Option<Identifier> {
        match self {
            Self::Loop(n)
            | Self::SwitchInLoop { loop_: n, .. }
            | Self::LoopInSwitch { loop_: n, .. } => Some(n.clone()),
            Self::None | Self::Switch(_) => None,
        }
    }
    fn switch_label(&self) -> Option<Identifier> {
        match self {
            Self::Switch(n)
            | Self::SwitchInLoop { switch: n, .. }
            | Self::LoopInSwitch { switch: n, .. } => Some(n.clone()),
            Self::None | Self::Loop(_) => None,
        }
    }
    fn into_switch(self, label: Identifier) -> Self {
        match self {
            Self::SwitchInLoop { loop_, .. }
            | Self::LoopInSwitch { loop_, .. }
            | Self::Loop(loop_) => Self::SwitchInLoop { loop_, switch: label },
            Self::Switch(_) | Self::None => Self::Switch(label),
        }
    }
    fn into_loop(self, label: Identifier) -> Self {
        match self {
            Self::SwitchInLoop { switch, .. }
            | Self::Switch(switch)
            | Self::LoopInSwitch { switch, .. } => Self::LoopInSwitch { loop_: label, switch },
            Self::None | Self::Loop(_) => Self::Loop(label),
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

    Break(Option<Identifier>),
    Continue(Option<Identifier>),
    While {
        cond: TypedExpr,
        body: Box<Stmt>,
        label: Option<Identifier>,
    },
    DoWhile {
        body: Box<Stmt>,
        cond: TypedExpr,
        label: Option<Identifier>,
    },
    For {
        init: Either<VarDecl, Option<TypedExpr>>,
        cond: Option<TypedExpr>,
        post: Option<TypedExpr>,

        body: Box<Stmt>,
        label: Option<Identifier>,
    },

    // extra credit
    GoTo(Identifier),
    Label(Identifier, Box<Stmt>),

    Switch {
        ctrl: TypedExpr,
        body: Box<Stmt>,
        label: Option<Identifier>,
        cases: Vec<Option<Const>>,
    },
    Case {
        cnst: TypedExpr,
        body: Box<Stmt>,
        label: Option<Identifier>,
    },
    Default {
        body: Box<Stmt>,
        label: Option<Identifier>,
    },

    Null,
}
impl Display for Stmt {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let indent = f.width().unwrap_or_default();
        let child = indent + 1;

        let pad = Identifier::from("\t").repeat(indent);

        let write_body = |body: &Self, f: &mut Formatter| match body {
            Self::Compound(_) => write!(f, "{body:indent$}"),
            _ => write!(f, "{body:child$}"),
        };

        match self {
            Self::Return(expr) => write!(f, "{pad}return  {expr}"),
            Self::Expression(expr) => write!(f, "{pad}{expr}"),
            Self::If { cond, then, else_ } => {
                writeln!(f, "{pad}if {cond}")?;
                write_body(then, f)?;
                if let Some(else_) = else_ {
                    writeln!(f, "{pad}else")?;
                    write_body(else_, f)
                } else {
                    Ok(())
                }
            }
            Self::Compound(block) => {
                for item in &block.0 {
                    writeln!(f, "{item:child$}")?;
                }
                Ok(())
            }

            Self::Break(_) => writeln!(f, "{pad}break"),
            Self::Continue(_) => writeln!(f, "{pad}continue"),
            Self::While { cond, body, .. } => {
                writeln!(f, "{pad}while {cond}")?;
                write_body(body, f)
            }
            Self::DoWhile { body, cond, .. } => {
                writeln!(f, "{pad}do")?;
                write_body(body, f)?;
                writeln!(f, "{pad}while {cond}")
            }
            Self::For { init, cond, post, body, .. } => {
                write!(f, "{pad}for ")?;

                match init {
                    Left(init) => write!(f, "{init}")?,
                    Right(Some(init)) => write!(f, "{init}")?,
                    Right(None) => write!(f, "---")?,
                }
                write!(f, "; ")?;
                match cond {
                    Some(cond) => write!(f, "{cond}")?,
                    None => write!(f, "---")?,
                }
                write!(f, "; ")?;
                match post {
                    Some(post) => write!(f, "{post}")?,
                    None => write!(f, "---")?,
                }
                writeln!(f)?;
                write_body(body, f)
            }
            Self::GoTo(name) => write!(f, "{pad}goto {name}"),
            Self::Label(name, body) => {
                writeln!(f, "{pad}label {name}:")?;
                write_body(body, f)
            }
            Self::Switch { ctrl, body, .. } => {
                writeln!(f, "{pad}switch {ctrl}:")?;
                write_body(body, f)
            }
            Self::Case { cnst, body, .. } => {
                writeln!(f, "{pad}case {cnst}:")?;
                write_body(body, f)
            }
            Self::Default { body, .. } => {
                writeln!(f, "{pad}default:")?;
                write_body(body, f)
            }
            Self::Null => Ok(()),
        }
    }
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

                Ok(Self::While { cond, body, label })
            }
            Self::DoWhile { cond, body, label } => {
                let cond = cond.resolve_identifiers(map)?;
                let body = Box::new(body.resolve_identifiers(map)?);

                Ok(Self::DoWhile { cond, body, label })
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

                Ok(Self::For { init, cond, post, body, label })
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
        func_name: Identifier,
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
                Ok(Self::Compound(block.resolve_goto_labels(labels, &func_name)?))
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

            // extra credit
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
                current_label.break_label().ok_or_else(|| anyhow::anyhow!("not a break label"))?,
            ))),
            Self::Continue(_) => Ok(Self::Continue(Some(
                current_label
                    .loop_label()
                    .ok_or_else(|| anyhow::anyhow!("not a continue label"))?,
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
            Self::Compound(block) => Ok(Self::Compound(block.resolve_loop_labels(&current_label)?)),
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
                    .type_
                    .ok_or_else(|| anyhow::anyhow!("switch type must be kbown at this point"))?;

                let mut switchctx = IndexSet::default();
                let body =
                    Box::new(body.resolve_switch_statements(Some(&mut switchctx), &switch_type)?);

                let cases = switchctx.into_iter().collect::<Vec<_>>();

                Self::Switch { ctrl, body, label, cases }
            }
            Self::Case { cnst, body, label } => {
                let switch_ctx =
                    switch_ctx.ok_or_else(|| anyhow::anyhow!("case outside of a switch"))?;

                let Expr::Const(value) = cnst.expr else {
                    anyhow::bail!("case value not a constant");
                };

                let value = value.cast_const(switch_type);
                let cnst = Expr::Const(value).typed(switch_type.clone());

                if !switch_ctx.insert(Some(value)) {
                    anyhow::bail!("case value already exists");
                };

                let body = Box::new(body.resolve_switch_statements(Some(switch_ctx), switch_type)?);
                Self::Case { cnst, body, label }
            }
            Self::Default { body, label } => {
                let switch_ctx =
                    switch_ctx.ok_or_else(|| anyhow::anyhow!("default outside of a switch"))?;

                if !switch_ctx.insert(None) {
                    anyhow::bail!("cdefault case already exists");
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
                let expr =
                    expr.type_check_and_convert(symbols)?.cast_by_assignment(enclosing_func_ret)?;

                Self::Return(expr)
            }
            Self::Expression(expr) => Self::Expression(expr.type_check_and_convert(symbols)?),
            Self::If { cond, then, else_ } => Self::If {
                cond: cond.type_check_and_convert(symbols)?,
                then: Box::new(then.type_check(symbols, enclosing_func_ret.clone())?),
                else_: match else_ {
                    Some(s) => Some(Box::new(s.type_check(symbols, enclosing_func_ret)?)),
                    None => None,
                },
            },
            Self::Compound(block) => {
                Self::Compound(block.type_check(symbols, &enclosing_func_ret)?)
            }
            Self::While { cond, body, label } => Self::While {
                cond: cond.type_check_and_convert(symbols)?,
                body: Box::new(body.type_check(symbols, enclosing_func_ret)?),
                label,
            },
            Self::DoWhile { body, cond, label } => Self::DoWhile {
                body: Box::new(body.type_check(symbols, enclosing_func_ret)?),
                cond: cond.type_check_and_convert(symbols)?,
                label,
            },
            Self::For { init, cond, post, body, label } => Self::For {
                init: match init {
                    Left(e) if e.sc != StorageClass::None => {
                        anyhow::bail!("specifier in for loop");
                    }
                    Left(d) => Left(d.type_check_block(symbols)?),
                    Right(Some(e)) => Right(Some(e.type_check_and_convert(symbols)?)),
                    Right(None) => Right(None),
                },
                cond: match cond {
                    Some(e) => Some(e.type_check_and_convert(symbols)?),
                    None => None,
                },
                post: match post {
                    Some(e) => Some(e.type_check_and_convert(symbols)?),
                    None => None,
                },
                body: Box::new(body.type_check(symbols, enclosing_func_ret)?),
                label,
            },
            Self::Label(label, stmt) => {
                Self::Label(label, Box::new(stmt.type_check(symbols, enclosing_func_ret)?))
            }
            Self::Switch { ctrl, body, label, cases } => {
                let ctrl = ctrl.type_check_and_convert(symbols)?;
                anyhow::ensure!(matches!(
                    ctrl.type_,
                    Some(Type::Int | Type::UInt | Type::Long | Type::ULong)
                ));

                Self::Switch {
                    ctrl,
                    body: Box::new(body.type_check(symbols, enclosing_func_ret)?),
                    label,
                    cases,
                }
            }
            Self::Case { cnst, body, label } => {
                let cnst = cnst.type_check_and_convert(symbols)?;
                anyhow::ensure!(matches!(
                    cnst.type_,
                    Some(Type::Int | Type::UInt | Type::Long | Type::ULong)
                ));
                Self::Case {
                    cnst,
                    body: Box::new(body.type_check(symbols, enclosing_func_ret)?),
                    label,
                }
            }
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
    pub type_: Option<Type>,
}
impl Display for TypedExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.expr)
    }
}
impl TypedExpr {
    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> anyhow::Result<Self> {
        Ok(Self { expr: self.expr.resolve_identifiers(map)?, type_: self.type_ })
    }

    fn type_check(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        self.expr.type_check(symbols)
    }
    fn type_check_and_convert(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<Self> {
        let res = self.expr.type_check(symbols)?;
        match res.type_.clone().unwrap() {
            Type::Array { element, .. } => {
                let addr = Expr::AddrOf(Box::new(res));
                Ok(addr.typed(Type::Pointer { to: Box::new(*element) }))
            }
            _ => Ok(res),
        }
    }

    fn is_place(&self) -> bool {
        self.expr.is_place()
    }
    fn is_null_pointer_constant(&self) -> bool {
        self.expr.is_null_pointer_constant()
    }
    fn common_ptr_type(&self, other: &Self) -> anyhow::Result<Type> {
        let e1_t = self.type_.clone().ok_or_else(|| anyhow::anyhow!("type should be known"))?;
        let e2_t = other.type_.clone().ok_or_else(|| anyhow::anyhow!("type should be known"))?;

        match () {
            _ if e1_t == e2_t => Ok(e1_t),
            _ if self.is_null_pointer_constant() => Ok(e2_t),
            _ if other.is_null_pointer_constant() => Ok(e1_t),
            _ => Err(anyhow::anyhow!("incompatible types")),
        }
    }

    fn cast_to_type(self, target: Type) -> Self {
        if self.type_ == Some(target.clone()) {
            return self;
        }
        Expr::Cast { target: target.clone(), inner: Box::new(self) }.typed(target)
    }

    fn cast_by_assignment(self, target: Type) -> anyhow::Result<Self> {
        if self.type_ == Some(target.clone()) {
            Ok(self)
        } else if (self.clone().type_.is_some_and(|t| t.is_arithmatic()) && target.is_arithmatic())
        // unsure about this
            || (self.is_null_pointer_constant() && matches!(target, Type::Pointer { .. }))
        {
            Ok(self.cast_to_type(target))
        } else {
            Err(anyhow::anyhow!("cannot convert type for assignment"))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Const(Const),
    Var(Identifier),
    Cast {
        target: Type,
        inner: Box<TypedExpr>,
    },
    Unary(UnaryOp, Box<TypedExpr>),
    Binary {
        op: BinaryOp,
        lhs: Box<TypedExpr>,
        rhs: Box<TypedExpr>,
    },
    CompoundAssignment {
        op: BinaryOp,
        lhs: Box<TypedExpr>,
        rhs: Box<TypedExpr>,
        common: Option<Type>,
    },
    Assignemnt(Box<TypedExpr>, Box<TypedExpr>),
    Conditional {
        cond: Box<TypedExpr>,
        then: Box<TypedExpr>,
        else_: Box<TypedExpr>,
    },
    FuncCall {
        name: Identifier,
        args: Vec<TypedExpr>,
    },
    Deref(Box<TypedExpr>),
    AddrOf(Box<TypedExpr>),
    Subscript(Box<TypedExpr>, Box<TypedExpr>),
}
impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Const(cnst) => write!(f, "{cnst}"),
            Self::Var(var) => write!(f, "{var}"),
            Self::Cast { target, inner } => write!(f, "({inner} as ({target}))"),
            Self::Unary(op, expr) => write!(f, "({op} {expr})"),
            Self::Binary { op, lhs, rhs } => write!(f, "({lhs} {op} {rhs})"),
            Self::CompoundAssignment { op, lhs, rhs, .. } => write!(f, "({lhs} {op}= {rhs})"),
            Self::Assignemnt(lhs, rhs) => write!(f, "({lhs} <- {rhs})"),
            Self::Conditional { cond, then, else_ } => {
                write!(f, "({cond} ? {then} : {else_})")
            }
            Self::FuncCall { name, args } => {
                let mut buf = Identifier::new();
                for (idx, arg) in args.iter().enumerate() {
                    write!(buf, "{}{arg}", if idx != 0 { ", " } else { "" })?;
                }
                write!(f, "{name}({buf})")
            }
            Self::AddrOf(expr) => write!(f, "(& {expr})"),
            Self::Deref(expr) => write!(f, "(* {expr})"),
            Self::Subscript(e1, e2) => write!(f, "({e1}[{e2}])"),
        }
    }
}
impl Expr {
    pub fn dummy_typed(self) -> TypedExpr {
        TypedExpr { expr: self, type_: None }
    }
    pub fn typed(self, ty: Type) -> TypedExpr {
        TypedExpr { expr: self, type_: Some(ty) }
    }
    fn is_place(&self) -> bool {
        matches!(self, Self::Var(_) | Self::Deref(_) | Self::Subscript(..))
    }
    fn is_null_pointer_constant(&self) -> bool {
        match self {
            Self::Const(c) => c.is_null_pointer_constant(),
            _ => false,
        }
    }

    fn resolve_identifiers(self, map: &mut Namespace<IdCtx>) -> anyhow::Result<Self> {
        Ok(match self {
            Self::Assignemnt(left, right) => Self::Assignemnt(
                Box::new(left.resolve_identifiers(map)?),
                Box::new(right.resolve_identifiers(map)?),
            ),

            Self::Unary(op, expr) => Self::Unary(op, Box::new(expr.resolve_identifiers(map)?)),
            Self::Binary { op, lhs, rhs } => Self::Binary {
                op,
                lhs: Box::new(lhs.resolve_identifiers(map)?),
                rhs: Box::new(rhs.resolve_identifiers(map)?),
            },
            Self::CompoundAssignment { op, lhs, rhs, common } => Self::CompoundAssignment {
                op,
                lhs: Box::new(lhs.resolve_identifiers(map)?),
                rhs: Box::new(rhs.resolve_identifiers(map)?),
                common,
            },

            v @ Self::Const(_) => v,
            Self::Conditional { cond, then, else_ } => Self::Conditional {
                cond: Box::new(cond.resolve_identifiers(map)?),
                then: Box::new(then.resolve_identifiers(map)?),
                else_: Box::new(else_.resolve_identifiers(map)?),
            },

            // magic happens here
            Self::Var(var) => map
                .get(&var)
                .cloned()
                .map(|t| Self::Var(t.name))
                .ok_or_else(|| anyhow::anyhow!("variable does not exist in map"))?,
            Self::FuncCall { name, args } => {
                let name = map
                    .get(&name)
                    .ok_or_else(|| anyhow::anyhow!("function does not exist in map"))?
                    .clone()
                    .name;
                let mut resolved_args = Vec::with_capacity(args.len());

                for arg in args {
                    resolved_args.push(arg.resolve_identifiers(map)?);
                }

                Self::FuncCall { name, args: resolved_args }
            }
            Self::Cast { target: to, inner: from } => {
                let from = Box::new(from.resolve_identifiers(map)?);
                Self::Cast { target: to, inner: from }
            }
            Self::AddrOf(expr) => Self::AddrOf(Box::new(expr.resolve_identifiers(map)?)),
            Self::Deref(expr) => Self::Deref(Box::new(expr.resolve_identifiers(map)?)),
            Self::Subscript(e1, e2) => Self::Subscript(
                Box::new(e1.resolve_identifiers(map)?),
                Box::new(e2.resolve_identifiers(map)?),
            ),
        })
    }

    fn type_check(self, symbols: &mut Namespace<TypeCtx>) -> anyhow::Result<TypedExpr> {
        match self {
            Self::AddrOf(expr) => {
                let expr = Box::new(expr.type_check(symbols)?);
                anyhow::ensure!(expr.is_place(), "cannot take the address of a non-place");

                let to = Box::new(
                    expr.type_
                        .clone()
                        .ok_or_else(|| anyhow::anyhow!("type must be known at this point"))?,
                );

                Ok(Self::AddrOf(expr).typed(Type::Pointer { to }))
            }
            Self::Deref(expr) => {
                let expr = expr.type_check_and_convert(symbols)?;
                if let Some(Type::Pointer { to }) = expr.type_.clone() {
                    Ok(Self::Deref(Box::new(expr)).typed(*to))
                } else {
                    anyhow::bail!("cannot deref a non-pointer")
                }
            }
            Self::FuncCall { name, args } => {
                let (types, ret) = match &symbols
                    .get(&name)
                    .ok_or_else(|| anyhow::anyhow!("function does not exist in symbol map"))?
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
                    let arg = arg.type_check_and_convert(symbols)?;
                    acc.push(arg.cast_by_assignment(ty)?);
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

            Self::Unary(op, expr) => {
                let expr = Box::new(expr.type_check_and_convert(symbols)?);

                let ty = match op {
                    UnaryOp::Complement if expr.type_ == Some(Type::Double) => {
                        anyhow::bail!("cannot complement a double")
                    }
                    UnaryOp::Complement | UnaryOp::Negate
                        if matches!(expr.type_, Some(Type::Pointer { .. })) =>
                    {
                        anyhow::bail!("cannot complement or negate a pointer")
                    }
                    UnaryOp::IncPre | UnaryOp::IncPost | UnaryOp::DecPre | UnaryOp::DecPost
                        if !expr.is_place() =>
                    {
                        anyhow::bail!("cannot increment or decrement a non place expression")
                    }
                    UnaryOp::Not => Type::Int,
                    _ => expr.clone().type_.expect("unary type should be known"),
                };

                Ok(Self::Unary(op, expr).typed(ty))
            }

            Self::Binary { op: op @ (BinaryOp::Equal | BinaryOp::NotEqual), lhs, rhs } => {
                let lhs = Box::new(lhs.type_check_and_convert(symbols)?);
                let rhs = Box::new(rhs.type_check_and_convert(symbols)?);

                let lhs_t = lhs.clone().type_.expect("type should exist");
                let rhs_t = rhs.clone().type_.expect("type should exist");

                let common_type = match (&lhs_t, &rhs_t) {
                    (Type::Pointer { .. }, _) | (_, Type::Pointer { .. }) => {
                        lhs.common_ptr_type(&rhs)?
                    }
                    _ => lhs_t.get_common_type(rhs_t),
                };

                let lhs = Box::new(lhs.cast_to_type(common_type.clone()));
                let rhs = Box::new(rhs.cast_to_type(common_type));

                Ok(Self::Binary { op, lhs, rhs }.typed(Type::Int))
            }

            Self::Binary { op, lhs, rhs } => {
                let lhs = Box::new(lhs.type_check_and_convert(symbols)?);
                let rhs = Box::new(rhs.type_check_and_convert(symbols)?);

                match op {
                    BinaryOp::And | BinaryOp::Or => {
                        return Ok(Self::Binary { op, lhs, rhs }.typed(Type::Int));
                    }
                    BinaryOp::Add | BinaryOp::Subtract => match (op, *lhs.clone(), *rhs.clone()) {
                        _ if lhs.clone().type_.unwrap().is_arithmatic()
                            && rhs.clone().type_.unwrap().is_arithmatic() => {}
                        // Pointer arithmatic
                        // how to check for null pointers? do i have to? todo
                        (BinaryOp::Add | BinaryOp::Subtract, ptr, other)
                        | (BinaryOp::Add, other, ptr)
                            if matches!(ptr.type_, Some(Type::Pointer { .. }))
                                && other.clone().type_.unwrap().is_intish() =>
                        {
                            let lhs = Box::new(ptr.clone());
                            let rhs = Box::new(other.cast_to_type(Type::Long));

                            return Ok(Self::Binary { op, lhs, rhs }.typed(ptr.type_.unwrap()));
                        }
                        (BinaryOp::Subtract, lhp, rhp)
                            if matches!(lhp.type_, Some(Type::Pointer { .. }))
                                && lhp.type_ == rhp.type_ =>
                        {
                            return Ok(Self::Binary { op, lhs: Box::new(lhp), rhs: Box::new(rhp) }
                                .typed(Type::Long));
                        }
                        _ => anyhow::bail!("invalid operators for arithmatic"),
                    },
                    BinaryOp::LessThan
                    | BinaryOp::LessOrEqual
                    | BinaryOp::GreaterThan
                    | BinaryOp::GreaterOrEqual => match (*lhs.clone(), *rhs.clone()) {
                        _ if lhs.clone().type_.unwrap().is_arithmatic()
                            && rhs.clone().type_.unwrap().is_arithmatic() => {}
                        (lhp, rhp)
                            if matches!(lhp.type_, Some(Type::Pointer { .. }))
                                && lhp.type_ == rhp.type_ =>
                        {
                            return Ok(Self::Binary { op, lhs: Box::new(lhp), rhs: Box::new(rhp) }
                                .typed(Type::Int));
                        }
                        _ => anyhow::bail!("invalid operators for comparison"),
                    },

                    BinaryOp::Reminder
                        if !(lhs.type_.as_ref().is_some_and(Type::is_intish)
                            && rhs.type_.as_ref().is_some_and(Type::is_intish)) =>
                    {
                        anyhow::bail!("invalid operands for modulo")
                    }
                    BinaryOp::Multiply | BinaryOp::Divide
                        if !(lhs.type_.as_ref().is_some_and(Type::is_arithmatic)
                            && rhs.type_.as_ref().is_some_and(Type::is_arithmatic)) =>
                    {
                        anyhow::bail!("invalid operands for multiplication")
                    }
                    _ => {}
                }

                let common = lhs
                    .clone()
                    .type_
                    .and_then(|lht| rhs.clone().type_.map(|rht| lht.get_common_type(rht)))
                    .expect("binary operand type should be known at this point");

                let lhs_cast = Box::new(lhs.clone().cast_to_type(common.clone()));
                let rhs = Box::new(rhs.cast_to_type(common.clone()));

                let ret = Self::Binary { op, lhs: lhs_cast, rhs: rhs.clone() };

                Ok(match op {
                    BinaryOp::Add
                    | BinaryOp::Subtract
                    | BinaryOp::Multiply
                    | BinaryOp::Divide
                    | BinaryOp::Reminder => ret.typed(common),
                    BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor
                        if rhs.type_.as_ref().is_some_and(Type::is_intish) =>
                    {
                        ret.typed(common)
                    }

                    BinaryOp::LeftShift | BinaryOp::RightShift
                        if rhs.type_.as_ref().is_some_and(Type::is_intish) =>
                    {
                        ret.typed(lhs.type_.unwrap())
                    }
                    BinaryOp::BitAnd
                    | BinaryOp::BitOr
                    | BinaryOp::BitXor
                    | BinaryOp::LeftShift
                    | BinaryOp::RightShift => {
                        anyhow::bail!("cannot apply operation to a double")
                    }

                    BinaryOp::GreaterThan
                    | BinaryOp::GreaterOrEqual
                    | BinaryOp::LessThan
                    | BinaryOp::LessOrEqual => ret.typed(Type::Int),

                    BinaryOp::Equal | BinaryOp::NotEqual | BinaryOp::And | BinaryOp::Or => {
                        unreachable!()
                    }
                })
            }

            Self::Assignemnt(lhs, rhs) => {
                let lhs = Box::new(lhs.type_check_and_convert(symbols)?);
                anyhow::ensure!(lhs.is_place(), "cannot assign to a non place expression");

                let rhs = rhs.type_check_and_convert(symbols)?;

                let left_type =
                    lhs.clone().type_.expect("assignee type should be known at this point");

                let rhs = Box::new(rhs.cast_by_assignment(left_type.clone())?);

                Ok(Self::Assignemnt(lhs, rhs).typed(left_type))
            }

            Self::CompoundAssignment { op, lhs, rhs, .. } => {
                // fix for pointer arithmatic todo
                let lhs = Box::new(lhs.type_check_and_convert(symbols)?);
                anyhow::ensure!(lhs.is_place(), "cannot assign to a non place expression");

                let rhs = Box::new(rhs.type_check_and_convert(symbols)?);

                let left_type =
                    lhs.clone().type_.expect("assignee type should be known at this point");

                let common_inner = left_type.clone().get_common_type(
                    rhs.type_
                        .clone()
                        .expect("compound assignment operand type should be known at this point"),
                );

                let common = match op {
                    BinaryOp::BitAnd
                    | BinaryOp::BitOr
                    | BinaryOp::BitXor
                    | BinaryOp::LeftShift
                    | BinaryOp::RightShift
                        if !common_inner.is_intish() =>
                    {
                        anyhow::bail!("can't operate those on a non integer")
                    }
                    BinaryOp::Reminder
                        if !(lhs.type_.as_ref().is_some_and(Type::is_intish)
                            && rhs.type_.as_ref().is_some_and(Type::is_intish)) =>
                    {
                        anyhow::bail!("cannot modulo a double or a pointer")
                    }
                    BinaryOp::Multiply | BinaryOp::Divide
                        if !(lhs.type_.as_ref().is_some_and(Type::is_arithmatic)
                            && rhs.type_.as_ref().is_some_and(Type::is_arithmatic)) =>
                    {
                        anyhow::bail!("cannot mulyiply a pointer")
                    }
                    BinaryOp::And | BinaryOp::Or => {
                        return Ok(Self::CompoundAssignment {
                            op,
                            lhs,
                            rhs,
                            common: Some(Type::Int),
                        }
                        .typed(Type::Int));
                    }

                    BinaryOp::Add
                    | BinaryOp::Subtract
                    | BinaryOp::Multiply
                    | BinaryOp::Divide
                    | BinaryOp::Reminder
                    | BinaryOp::BitAnd
                    | BinaryOp::BitOr
                    | BinaryOp::BitXor => Some(common_inner.clone()),

                    BinaryOp::LeftShift | BinaryOp::RightShift => Some(left_type.clone()),

                    BinaryOp::Equal
                    | BinaryOp::NotEqual
                    | BinaryOp::GreaterThan
                    | BinaryOp::GreaterOrEqual
                    | BinaryOp::LessThan
                    | BinaryOp::LessOrEqual => unreachable!(),
                };

                let rhs = Box::new(rhs.cast_to_type(common_inner));

                Ok(Self::CompoundAssignment { op, lhs, rhs, common }.typed(left_type))
            }
            Self::Const(c) => match c {
                Const::Int(_) => Ok(self.typed(Type::Int)),
                Const::Long(_) => Ok(self.typed(Type::Long)),
                Const::UInt(_) => Ok(self.typed(Type::UInt)),
                Const::ULong(_) => Ok(self.typed(Type::ULong)),
                Const::Double(_) => Ok(self.typed(Type::Double)),
            },
            Self::Conditional { cond, then, else_ } => {
                let cond = Box::new(cond.type_check_and_convert(symbols)?);
                let then = then.type_check_and_convert(symbols)?;
                let else_ = else_.type_check_and_convert(symbols)?;

                let then_t = then.clone().type_.expect("then type must be known");
                let else_t = else_.clone().type_.expect("then type must be known");

                let common = match (&then_t, &else_t) {
                    (Type::Pointer { .. }, _) | (_, Type::Pointer { .. }) => {
                        then.common_ptr_type(&else_)?
                    }
                    _ => then_t.get_common_type(else_t),
                };

                let then = Box::new(then.cast_to_type(common.clone()));
                let else_ = Box::new(else_.cast_to_type(common.clone()));

                Ok(Self::Conditional { cond, then, else_ }.typed(common))
            }

            Self::Cast { target, inner } => {
                let inner = Box::new(inner.type_check_and_convert(symbols)?);
                match (&target, inner.type_.as_ref().expect("type must be known")) {
                    (Type::Double, Type::Pointer { .. }) | (Type::Pointer { .. }, Type::Double) => {
                        anyhow::bail!("cannot cast pointers to double or vice versa")
                    }
                    (Type::Array { .. }, _) => anyhow::bail!("cannot cast to arrays"),
                    _ => {}
                }
                Ok(Self::Cast { target: target.clone(), inner }.typed(target))
            }
            Self::Subscript(fst, snd) => {
                let fst = fst.type_check_and_convert(symbols)?;
                let snd = snd.type_check_and_convert(symbols)?;

                let (ptr, int, ptr_type) = match (fst, snd) {
                    // a[0] amd 0[a] are both valid for some reason.
                    (ptr, other) | (other, ptr)
                        if matches!(ptr.type_.as_ref().unwrap(), Type::Pointer { .. })
                            && other.type_.as_ref().unwrap().is_intish() =>
                    {
                        let other = other.cast_to_type(Type::Long);
                        let Some(Type::Pointer { to }) = ptr.type_.clone() else { unreachable!() };
                        (ptr, other, to)
                    }
                    _ => anyhow::bail!("subscript must have integer and pointer operands"),
                };

                Ok(Self::Subscript(Box::new(ptr), Box::new(int)).typed(*ptr_type))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Const {
    Int(i32),
    Long(i64),
    UInt(u32),
    ULong(u64),
    Double(f64),
}
// Const values do not include NaN.
impl Eq for Const {}
// Hash is used in switch statements, which cannot be doubles.
impl Hash for Const {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Int(i) => i.hash(state),
            Self::Long(i) => i.hash(state),
            Self::UInt(i) => i.hash(state),
            Self::ULong(i) => i.hash(state),
            Self::Double(_) => {}
        }
    }
}

impl Display for Const {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // this is used for assembly generation for some reason.
        match self {
            Self::Int(i) => write!(f, "{i}"),
            Self::Long(i) => write!(f, "{i}"),
            Self::UInt(i) => write!(f, "{i}"),
            Self::ULong(i) => write!(f, "{i}"),

            // maybe?
            Self::Double(i) => write!(f, "{i:.3}"),
        }
    }
}

macro_rules! const_cast {
    ($self:ident, $target_type:ident, [$($types:tt),+ $(,)?]) => {

        match ($self, $target_type) {
            (_, Type::Func { .. }) => unreachable!(),
            (Self::Int(0), Type::Pointer{..}) => Self::ULong(0),
            (Self::Long(0), Type::Pointer{..}) => Self::ULong(0),
            (Self::UInt(0), Type::Pointer{..}) => Self::ULong(0),
            (Self::ULong(0), Type::Pointer{..}) => Self::ULong(0),

            $(
                (Self::Int(i), Type::$types) => Self::$types(i as _),
                (Self::Long(i), Type::$types) => Self::$types(i as _),
                (Self::UInt(i), Type::$types) => Self::$types(i as _),
                (Self::ULong(i), Type::$types) => Self::$types(i as _),

                // maybe?
                (Self::Double(i), Type::$types) => Self::$types(i as _),
            )+

            _ => unreachable!("Debug: Unhandled const cast from {:?} to {:?}", $self, $target_type),
        }
    };
}

impl Const {
    fn cast_const(self, target_type: &Type) -> Self {
        const_cast!(self, target_type, [Int, Long, UInt, ULong, Double])
    }
    fn into_static_init(self, target_type: &Type) -> StaticInit {
        let matching_const = self.cast_const(target_type);
        match matching_const {
            Self::Int(i) => StaticInit::Int(i),
            Self::Long(i) => StaticInit::Long(i),
            Self::UInt(i) => StaticInit::UInt(i),
            Self::ULong(i) => StaticInit::ULong(i),
            Self::Double(i) => StaticInit::Double(i),
        }
    }
    fn is_null_pointer_constant(&self) -> bool {
        matches!(self, Self::Int(0) | Self::Long(0) | Self::UInt(0) | Self::ULong(0))
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

    // chapter 14 pointers
    AddressOf,
    Dereference,
}
impl Display for UnaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Complement => f.pad("~"),
            Self::Negate => f.pad("-"),
            Self::Not => f.pad("!"),
            Self::Plus => f.pad(""),
            Self::IncPre => f.pad("++@"),
            Self::IncPost => f.pad("@++"),
            Self::DecPre => f.pad("--@"),
            Self::DecPost => f.pad("@--"),
            Self::AddressOf => f.pad("&"),
            Self::Dereference => f.pad("*"),
        }
    }
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
impl Display for BinaryOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => f.pad("+"),
            Self::Subtract => f.pad("-"),
            Self::Multiply => f.pad("*"),
            Self::Divide => f.pad("/"),
            Self::Reminder => f.pad("%"),
            Self::And => f.pad("&&"),
            Self::Or => f.pad("||"),
            Self::Equal => f.pad("=="),
            Self::NotEqual => f.pad("!="),
            Self::LessThan => f.pad("<"),
            Self::LessOrEqual => f.pad("<="),
            Self::GreaterThan => f.pad(">"),
            Self::GreaterOrEqual => f.pad(">="),
            Self::BitAnd => f.pad("&"),
            Self::BitOr => f.pad("|"),
            Self::BitXor => f.pad("^"),
            Self::LeftShift => f.pad("<<"),
            Self::RightShift => f.pad(">>"),
        }
    }
}

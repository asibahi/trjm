use crate::ast::{self, Const, Namespace, StaticInit, StorageClass, Type, TypeCtx};
use ecow::{EcoString as Identifier, eco_format};
use either::Either::{Left, Right};
use std::{
    cmp::Ordering,
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

pub static GEN: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct Program {
    pub top_level: Vec<TopLevel>,
    pub symbols: Namespace<TypeCtx>,
}
impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Program:")?;
        writeln!(f, "\tSymbols:")?;

        for (name, type_ctx) in &self.symbols {
            writeln!(f, "\t\t{name:<8} {type_ctx}")?;
        }
        writeln!(f)?;
        for tl in &self.top_level {
            writeln!(f, "\t{tl}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum TopLevel {
    Function { name: Identifier, global: bool, params: Vec<Identifier>, body: Vec<Instr> },
    StaticVar { name: Identifier, global: bool, type_: Type, init: Vec<StaticInit> },
}
impl Display for TopLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StaticVar { name, global, type_, init } => {
                writeln!(
                    f,
                    "Static: {name}. global:{global}. type: {type_}. initial value: {init:?}"
                )
            }
            Self::Function { name, global, params, body } => {
                write!(f, "Function: {name}. global:{global}. (")?;
                for (idx, param) in params.iter().enumerate() {
                    write!(f, "{}{param}", if idx != 0 { ", " } else { "" })?;
                }
                writeln!(f, ")")?;

                for instr in body {
                    writeln!(f, "\t\t{instr}")?;
                }

                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Instr {
    Return(Value),
    Truncate { src: Value, dst: Place },
    SignExtend { src: Value, dst: Place },
    ZeroExtend { src: Value, dst: Place },
    DoubleToInt { src: Value, dst: Place },
    DoubleToUInt { src: Value, dst: Place },
    IntToDouble { src: Value, dst: Place },
    UIntToDouble { src: Value, dst: Place },

    Unary { op: UnOp, src: Value, dst: Place },
    Binary { op: BinOp, lhs: Value, rhs: Value, dst: Place },
    Copy { src: Value, dst: Place },

    GetAddress { src: Value, dst: Place },
    Load { src_ptr: Value, dst: Place },
    Store { src: Value, dst_ptr: Place },

    Jump { target: Identifier },
    JumpIfZero { cond: Value, target: Identifier },
    JumpIfNotZero { cond: Value, target: Identifier },
    Label(Identifier),
    FuncCall { name: Identifier, args: Vec<Value>, dst: Place },

    AddPtr { ptr: Value, idx: Value, scale: usize, dst: Place },
    CopyToOffset { src: Value, dst: Identifier, offset: usize },
}
impl Display for Instr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Return(value) => write!(f, "return {value}"),
            Self::Truncate { src, dst } => write!(f, "{:<8} <- truncate {src}", dst.0),
            Self::SignExtend { src, dst } => write!(f, "{:<8} <- sign_extend {src}", dst.0),
            Self::ZeroExtend { src, dst } => write!(f, "{:<8} <- zero_extend {src}", dst.0),

            Self::DoubleToInt { src, dst } => write!(f, "{:<8} <- dbl_to_int {src}", dst.0),
            Self::DoubleToUInt { src, dst } => write!(f, "{:<8} <- dbl_to_uint {src}", dst.0),
            Self::IntToDouble { src, dst } => write!(f, "{:<8} <- int_to_dbl {src}", dst.0),
            Self::UIntToDouble { src, dst } => write!(f, "{:<8} <- uint_to_dbl {src}", dst.0),

            Self::Unary { op, src, dst } => write!(f, "{:<8} <- {op} {src}", dst.0),
            Self::Binary { op, lhs, rhs, dst } => write!(f, "{:<8} <- {lhs} {op} {rhs}", dst.0),
            Self::Copy { src, dst } => write!(f, "{:<8} <- copy {src}", dst.0),
            Self::Jump { target } => write!(f, "jump     {:<10} -> {target}", ""),
            Self::JumpIfZero { cond, target } => write!(f, "jump_z   {cond:<10} -> {target}"),
            Self::JumpIfNotZero { cond, target } => write!(f, "jump_nz  {cond:<10} -> {target}"),
            Self::Label(target) => write!(f, "\tLABEL {target}"),
            Self::FuncCall { name, args, dst } => {
                write!(f, "{:<8} <- call {name}(", dst.0)?;
                for (idx, arg) in args.iter().enumerate() {
                    write!(f, "{}{arg}", if idx != 0 { ", " } else { "" })?;
                }
                write!(f, ")")
            }

            Self::GetAddress { src, dst } => write!(f, "{:<8} <- addr {}", dst.0, src),
            Self::Load { src_ptr, dst } => write!(f, "{:<8} <- load {src_ptr}", dst.0),
            Self::Store { src, dst_ptr } => write!(f, "{:<8} <- store {src}", dst_ptr.0),

            Self::AddPtr { ptr, idx, scale, dst } => {
                write!(f, "{:<8} <- {ptr} ++ ({idx} * {scale})", dst.0)
            }
            Self::CopyToOffset { src, dst, offset } => write!(f, "{:<8} <- {src} @ {offset}", dst),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Const(Const),
    Var(Place),
}
impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Const(cnst) => write!(f, "({cnst})"),
            Self::Var(place) => write!(f, "[{}]", place.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Place(pub Identifier);

#[derive(Debug, Clone, Copy)]
pub enum UnOp {
    Complement,
    Negate,
    Not,
}
impl Display for UnOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Complement => f.pad("~"),
            Self::Negate => f.pad("-"),
            Self::Not => f.pad("!"),
        }
    }
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
impl Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => f.pad("+"),
            Self::Subtract => f.pad("-"),
            Self::Multiply => f.pad("*"),
            Self::Divide => f.pad("/"),
            Self::Reminder => f.pad("%"),
            Self::BitAnd => f.pad("&"),
            Self::BitOr => f.pad("|"),
            Self::BitXor => f.pad("^"),
            Self::LeftShift => f.pad("<<"),
            Self::RightShift => f.pad(">>"),
            Self::Equal => f.pad("=="),
            Self::NotEqual => f.pad("!="),
            Self::LessThan => f.pad("<"),
            Self::LessOrEqual => f.pad("<="),
            Self::GreaterThan => f.pad(">"),
            Self::GreaterOrEqual => f.pad(">="),
        }
    }
}

// ======

impl ast::Program {
    pub fn to_ir(&self, instrs: &mut Vec<Instr>, symbols: &mut Namespace<TypeCtx>) -> Program {
        //
        // traverse AST
        let mut top_level = self
            .decls
            .iter()
            .filter_map(|decl| -> Option<TopLevel> {
                match decl {
                    ast::Decl::Func(fun) => fun.body.as_ref().map(|_| fun.to_ir(instrs, symbols)),
                    ast::Decl::Var(_) => None,
                }
            })
            .collect::<Vec<_>>();

        // traverse symbol table
        top_level.extend(symbols.iter().filter_map(|(name, type_ctx)| match &type_ctx.attr {
            ast::Attributes::Static { init, global } => match init {
                ast::Tentative => Some(TopLevel::StaticVar {
                    name: name.clone(),
                    global: *global,
                    init: vec![type_ctx.type_.zeroed_static()], // placeholder. todo
                    type_: type_ctx.type_.clone(),
                }),
                ast::Initial(init) => Some(TopLevel::StaticVar {
                    name: name.clone(),
                    global: *global,
                    init: init.clone(), // placeholder. todo
                    type_: type_ctx.type_.clone(),
                }),

                ast::NoInit => None,
            },
            _ => None,
        }));

        Program { top_level, symbols: std::mem::take(symbols) }
    }
}

fn make_ir_variable(prefix: &'static str, type_: Type, symbols: &mut Namespace<TypeCtx>) -> Place {
    let tmp_name = eco_format!("{prefix}.{}", GEN.fetch_add(1, Relaxed));
    symbols.insert(tmp_name.clone(), TypeCtx { type_, attr: ast::Attributes::Local });
    Place(tmp_name)
}

impl ast::FuncDecl {
    fn to_ir(&self, instrs: &mut Vec<Instr>, symbols: &mut Namespace<TypeCtx>) -> TopLevel {
        let Some(ref body) = self.body else { unreachable!() };
        body.to_ir(instrs, symbols);

        if !matches!(instrs.last(), Some(Instr::Return(_))) {
            instrs.push(Instr::Return(Value::Const(Const::Int(0))));
        }

        TopLevel::Function {
            name: self.name.clone(),
            params: self.params.clone(),
            body: std::mem::take(instrs),
            global: self.sc == StorageClass::Extern,
        }
    }
}
impl ast::Stmt {
    fn to_ir(&self, instrs: &mut Vec<Instr>, symbols: &mut Namespace<TypeCtx>) {
        let goto_label = |s| eco_format!("goto.{s}");

        let brk_label = |s| eco_format!("brk.{s}");
        let cntn_label = |s| eco_format!("cntn.{s}");

        let case_label = |s, i| eco_format!("case.{i}.{s}").replace("-", "n");
        let dfl_label = |s| eco_format!("dflt.{s}");

        match self {
            Self::Return(expr) => {
                let dst = expr.to_ir_and_convert(instrs, symbols);
                instrs.push(Instr::Return(dst));
            }
            Self::Expression(expr) => {
                expr.to_ir(instrs, symbols);
            }
            Self::If { cond, then, else_ } => {
                let counter = GEN.fetch_add(1, Relaxed);

                let else_label = eco_format!("else.{}", counter);

                let cond = cond.to_ir_and_convert(instrs, symbols);
                // Do I need Copy Instr here ?
                instrs.push(Instr::JumpIfZero { cond, target: else_label.clone() });

                then.to_ir(instrs, symbols);

                if let Some(else_) = else_ {
                    let end_label = eco_format!("end.{}", counter);

                    instrs.extend([
                        Instr::Jump { target: end_label.clone() },
                        Instr::Label(else_label),
                    ]);

                    else_.to_ir(instrs, symbols);

                    instrs.push(Instr::Label(end_label));
                } else {
                    instrs.push(Instr::Label(else_label));
                }
            }

            Self::Compound(b) => b.to_ir(instrs, symbols),

            Self::GoTo(label) => {
                instrs.push(Instr::Jump { target: goto_label(label) });
            }
            Self::Label(label, stmt) => {
                instrs.push(Instr::Label(goto_label(label)));
                stmt.to_ir(instrs, symbols);
            }

            Self::Null => {}

            Self::Switch { ctrl, body, label: Some(label), cases } => {
                let dst = make_ir_variable(
                    "swch",
                    ctrl.type_.clone().expect("switch type must be known"),
                    symbols,
                );

                let ctrl = ctrl.to_ir_and_convert(instrs, symbols);

                for v in cases.iter().filter_map(|c| *c) {
                    instrs.extend([
                        Instr::Binary {
                            op: BinOp::Equal,
                            lhs: Value::Const(v),
                            rhs: ctrl.clone(),
                            dst: dst.clone(),
                        },
                        Instr::JumpIfNotZero {
                            cond: Value::Var(dst.clone()),
                            target: case_label(label, v),
                        },
                    ]);
                }
                if cases.contains(&None) {
                    instrs.push(Instr::Jump { target: dfl_label(label) });
                } else {
                    instrs.push(Instr::Jump { target: brk_label(label) });
                }

                body.to_ir(instrs, symbols);
                instrs.push(Instr::Label(brk_label(label)));
            }
            Self::Case { cnst, body, label: Some(label) } => {
                let ast::Expr::Const(value) = cnst.expr else { unreachable!() };
                instrs.push(Instr::Label(case_label(label, value)));

                body.to_ir(instrs, symbols);
            }
            Self::Default { body, label: Some(label) } => {
                instrs.push(Instr::Label(dfl_label(label)));

                body.to_ir(instrs, symbols);
            }

            Self::Break(Some(label)) => {
                instrs.push(Instr::Jump { target: brk_label(label) });
            }
            Self::Continue(Some(label)) => {
                instrs.push(Instr::Jump { target: cntn_label(label) });
            }
            Self::DoWhile { body, cond, label: Some(label) } => {
                let start_label = eco_format!("strt.{label}");

                instrs.push(Instr::Label(start_label.clone()));
                body.to_ir(instrs, symbols);

                instrs.push(Instr::Label(cntn_label(label)));
                let cond = cond.to_ir_and_convert(instrs, symbols);

                instrs.extend([
                    Instr::JumpIfNotZero { cond, target: start_label },
                    Instr::Label(brk_label(label)),
                ]);
            }
            Self::While { cond, body, label: Some(label) } => {
                let start_label = cntn_label(label);
                let end_label = brk_label(label);
                instrs.push(Instr::Label(start_label.clone()));

                let cond = cond.to_ir_and_convert(instrs, symbols);
                instrs.push(Instr::JumpIfZero { cond, target: end_label.clone() });
                body.to_ir(instrs, symbols);

                instrs.extend([Instr::Jump { target: start_label }, Instr::Label(end_label)]);
            }
            Self::For { init, cond, post, body, label: Some(label) } => {
                let start_label = eco_format!("strt.{label}");
                let cntn_label = cntn_label(label);
                let brk_label = brk_label(label);

                match init {
                    Left(decl) => decl.to_ir(instrs, symbols),
                    Right(Some(expr)) => drop(expr.to_ir(instrs, symbols)),
                    Right(None) => (),
                }

                instrs.push(Instr::Label(start_label.clone()));

                if let Some(cond) = cond {
                    let cond = cond.to_ir_and_convert(instrs, symbols);
                    instrs.push(Instr::JumpIfZero { cond, target: brk_label.clone() });
                }

                body.to_ir(instrs, symbols);

                instrs.push(Instr::Label(cntn_label));
                if let Some(post) = post {
                    post.to_ir(instrs, symbols);
                }

                instrs.extend([Instr::Jump { target: start_label }, Instr::Label(brk_label)]);
            }
            any => unreachable!("statement shouldn't exist: {any:?}"),
        }
    }
}

enum ExprResult {
    Plain(Value),
    DerefPtr(Value),
}

impl ast::TypedExpr {
    fn to_ir_and_convert(
        &self,
        instrs: &mut Vec<Instr>,
        symbols: &mut Namespace<TypeCtx>,
    ) -> Value {
        self.expr.to_ir_and_convert(
            instrs,
            symbols,
            self.type_.as_ref().expect("expr type is known"),
        )
    }
    fn to_ir(&self, instrs: &mut Vec<Instr>, symbols: &mut Namespace<TypeCtx>) -> ExprResult {
        self.expr.to_ir(instrs, symbols, self.type_.as_ref().expect("expr type is known"))
    }
}

impl ast::Expr {
    fn to_ir_and_convert(
        &self,
        instrs: &mut Vec<Instr>,
        symbols: &mut Namespace<TypeCtx>,
        expr_type: &Type,
    ) -> Value {
        let result = self.to_ir(instrs, symbols, expr_type);
        match result {
            ExprResult::Plain(value) => value,
            ExprResult::DerefPtr(ptr) => {
                let dst = make_ir_variable("ptr", expr_type.clone(), symbols);
                instrs.push(Instr::Load { src_ptr: ptr, dst: dst.clone() });

                Value::Var(dst)
            }
        }
    }

    fn to_ir(
        &self,
        instrs: &mut Vec<Instr>,
        symbols: &mut Namespace<TypeCtx>,
        expr_type: &Type,
    ) -> ExprResult {
        match self {
            Self::Const(i) => ExprResult::Plain(Value::Const(*i)),

            Self::Unary(ast::UnaryOp::Plus, expr) => expr.to_ir(instrs, symbols),
            Self::Unary(
                op @ (ast::UnaryOp::IncPost
                | ast::UnaryOp::DecPost
                | ast::UnaryOp::IncPre
                | ast::UnaryOp::DecPre),
                expr,
            ) => postfix_prefix_instrs(instrs, symbols, *op, expr, expr_type),
            Self::Unary(unary_op, expr) => {
                let src = expr.to_ir_and_convert(instrs, symbols);

                let dst = make_ir_variable("uop", expr_type.clone(), symbols);
                let op = unary_op.to_ir();
                instrs.push(Instr::Unary { op, src, dst: dst.clone() });

                ExprResult::Plain(Value::Var(dst))
            }
            Self::Binary { op: op @ (ast::BinaryOp::And | ast::BinaryOp::Or), lhs, rhs } => {
                logical_ops_instrs(instrs, symbols, *op, lhs, rhs, expr_type)
            }
            Self::Binary { op: op @ (ast::BinaryOp::Add | ast::BinaryOp::Subtract), lhs, rhs }
                if matches!(lhs.type_, Some(Type::Pointer { .. }))
                    && rhs.type_.as_ref().is_some_and(|t| t.is_intish()) =>
            {
                let Some(Type::Pointer { ref to }) = lhs.type_ else { unreachable!() };
                let lhs = lhs.to_ir_and_convert(instrs, symbols);
                let rhs = match op {
                    ast::BinaryOp::Add => rhs.to_ir_and_convert(instrs, symbols),
                    ast::BinaryOp::Subtract => Self::Unary(ast::UnaryOp::Negate, rhs.clone())
                        .to_ir_and_convert(instrs, symbols, rhs.type_.as_ref().unwrap()),
                    _ => unreachable!(),
                };

                let dst = make_ir_variable("sbs", expr_type.clone(), symbols);

                instrs.push(Instr::AddPtr {
                    ptr: lhs,
                    idx: rhs,
                    scale: to.size(),
                    dst: dst.clone(),
                });
                ExprResult::Plain(Value::Var(dst))
            }
            Self::Binary { op: ast::BinaryOp::Subtract, lhs, rhs }
                if matches!(lhs.type_, Some(Type::Pointer { .. }))
                    && matches!(rhs.type_, Some(Type::Pointer { .. })) =>
            {
                let Some(Type::Pointer { ref to }) = lhs.type_ else { unreachable!() };
                let lhs = lhs.to_ir_and_convert(instrs, symbols);
                let rhs = rhs.to_ir_and_convert(instrs, symbols);

                let diff = make_ir_variable("diff", expr_type.clone(), symbols);
                let dst = make_ir_variable("psub", expr_type.clone(), symbols);

                instrs.extend([
                    Instr::Binary { op: BinOp::Subtract, lhs, rhs, dst: diff.clone() },
                    Instr::Binary {
                        op: BinOp::Divide,
                        lhs: Value::Var(diff),
                        rhs: Value::Const(Const::ULong(to.size() as _)),
                        dst: dst.clone(),
                    },
                ]);

                ExprResult::Plain(Value::Var(dst))
            }
            Self::Binary { op, lhs, rhs } => {
                let lhs = lhs.to_ir_and_convert(instrs, symbols);
                let rhs = rhs.to_ir_and_convert(instrs, symbols);

                let dst = make_ir_variable("bop", expr_type.clone(), symbols);

                let op = op.to_ir();

                instrs.push(Instr::Binary { op, lhs, rhs, dst: dst.clone() });
                ExprResult::Plain(Value::Var(dst))
            }
            Self::Var(id) => ExprResult::Plain(Value::Var(Place(id.clone()))),
            Self::Assignemnt(place, value) => {
                let value = if place.type_ == value.type_ {
                    value.to_ir_and_convert(instrs, symbols)
                } else {
                    Self::Cast { target: place.type_.clone().unwrap(), inner: value.clone() }
                        .typed(place.type_.clone().unwrap())
                        .to_ir_and_convert(instrs, symbols)
                };

                let place = place.to_ir(instrs, symbols);

                match &place {
                    ExprResult::Plain(obj) => {
                        let Value::Var(obj) = obj else { unreachable!() };
                        instrs.push(Instr::Copy { src: value, dst: obj.clone() });

                        place
                    }
                    ExprResult::DerefPtr(ptr) => {
                        let Value::Var(ptr) = ptr else {
                            unreachable!("pointer cannot be a constant")
                        };
                        instrs.push(Instr::Store { src: value.clone(), dst_ptr: ptr.clone() });
                        ExprResult::Plain(value)
                    }
                }
            }

            Self::CompoundAssignment { op, lhs, rhs, common } => {
                let common = common.clone().unwrap();
                let cast_op = common != *expr_type;

                let lhs = lhs.to_ir(instrs, symbols);
                let rhs = rhs.to_ir_and_convert(instrs, symbols);
                let tmp = make_ir_variable("cmpd", expr_type.clone(), symbols);

                match &lhs {
                    ExprResult::Plain(obj) => {
                        let Value::Var(dst) = obj else { unreachable!() };

                        let obj = if cast_op {
                            let obj =
                                emit_cast_instr(instrs, symbols, &common, obj.clone(), expr_type);
                            Value::Var(obj)
                        } else {
                            obj.clone()
                        };
                        instrs.push(Instr::Binary {
                            op: op.to_ir(),
                            lhs: obj,
                            rhs,
                            dst: tmp.clone(),
                        });
                        let tmp = if cast_op {
                            emit_cast_instr(instrs, symbols, expr_type, Value::Var(tmp), &common)
                        } else {
                            tmp
                        };
                        instrs.push(Instr::Copy { src: Value::Var(tmp), dst: dst.clone() });

                        lhs
                    }
                    ExprResult::DerefPtr(ptr) => {
                        let Value::Var(dst) = ptr else {
                            unreachable!("pointer cannot be a constant")
                        };

                        let lhs = make_ir_variable("drf", expr_type.clone(), symbols);
                        instrs.push(Instr::Load { src_ptr: ptr.clone(), dst: lhs.clone() });

                        let lhs = Value::Var(if cast_op {
                            emit_cast_instr(instrs, symbols, &common, Value::Var(lhs), expr_type)
                        } else {
                            lhs
                        });

                        instrs.push(Instr::Binary { op: op.to_ir(), lhs, rhs, dst: tmp.clone() });

                        let tmp = if cast_op {
                            emit_cast_instr(instrs, symbols, expr_type, Value::Var(tmp), &common)
                        } else {
                            tmp
                        };
                        let ret = Value::Var(tmp);
                        instrs.push(Instr::Store { src: ret.clone(), dst_ptr: dst.clone() });
                        ExprResult::Plain(ret)
                    }
                }
            }

            Self::Conditional { cond, then, else_ } => {
                let counter = GEN.fetch_add(1, Relaxed);

                let end_label = eco_format!("end.{}", counter);
                let e2_label = eco_format!("e2.{}", counter);

                let result = make_ir_variable(
                    "ter",
                    then.type_.clone().expect("ternary type should be known"),
                    symbols,
                );

                let cond = cond.to_ir_and_convert(instrs, symbols);
                instrs.push(Instr::JumpIfZero { cond, target: e2_label.clone() });

                let v1 = then.to_ir_and_convert(instrs, symbols);
                instrs.extend([
                    Instr::Copy { src: v1, dst: result.clone() },
                    Instr::Jump { target: end_label.clone() },
                    Instr::Label(e2_label),
                ]);

                let v2 = else_.to_ir_and_convert(instrs, symbols);
                instrs.extend([
                    Instr::Copy { src: v2, dst: result.clone() },
                    Instr::Label(end_label),
                ]);

                ExprResult::Plain(Value::Var(result))
            }

            Self::FuncCall { name, args } => {
                let dst = make_ir_variable("cll", expr_type.clone(), symbols);

                let vals = args.iter().map(|arg| arg.to_ir_and_convert(instrs, symbols)).collect();

                instrs.push(Instr::FuncCall { name: name.clone(), args: vals, dst: dst.clone() });

                ExprResult::Plain(Value::Var(dst))
            }

            Self::Cast { target, inner } => {
                let src = inner.to_ir_and_convert(instrs, symbols);
                let src_ty = inner.type_.as_ref().expect("inner expr type should be known");
                if target == src_ty {
                    return ExprResult::Plain(src);
                }

                let dst_var = emit_cast_instr(instrs, symbols, target, src, src_ty);

                ExprResult::Plain(Value::Var(dst_var))
            }
            Self::AddrOf(inner) => {
                let v = inner.to_ir(instrs, symbols);
                match v {
                    ExprResult::Plain(obj) => {
                        let dst = make_ir_variable("addr", expr_type.clone(), symbols);
                        instrs.push(Instr::GetAddress { src: obj, dst: dst.clone() });
                        ExprResult::Plain(Value::Var(dst))
                    }
                    ExprResult::DerefPtr(value) => ExprResult::Plain(value),
                }
            }
            Self::Deref(inner) => {
                let result = inner.to_ir_and_convert(instrs, symbols);
                ExprResult::DerefPtr(result)
            }
            Self::Subscript(lhs, rhs) => {
                // after type checking, ptr is always first
                let ExprResult::Plain(res) =
                    Self::Binary { op: ast::BinaryOp::Add, lhs: lhs.clone(), rhs: rhs.clone() }
                        .to_ir(instrs, symbols, expr_type)
                else {
                    unreachable!()
                };
                ExprResult::DerefPtr(res)
            }
        }
    }
}

fn emit_cast_instr(
    instrs: &mut Vec<Instr>,
    symbols: &mut Namespace<TypeCtx>,
    target: &Type,
    src: Value,
    src_ty: &Type,
) -> Place {
    let dst_var = make_ir_variable("cst", target.clone(), symbols);
    let dst = dst_var.clone();

    let cast_instr = match (target, src_ty) {
        (Type::Int | Type::Long, Type::Double) => Instr::DoubleToInt { src, dst },
        (Type::UInt | Type::ULong, Type::Double) => Instr::DoubleToUInt { src, dst },
        (Type::Double, Type::Int | Type::Long) => Instr::IntToDouble { src, dst },
        (Type::Double, Type::UInt | Type::ULong) => Instr::UIntToDouble { src, dst },

        _ => match target.size().cmp(&src_ty.size()) {
            Ordering::Equal => Instr::Copy { src, dst },
            Ordering::Less => Instr::Truncate { src, dst },
            _ if src_ty.signed() => Instr::SignExtend { src, dst },
            _ => Instr::ZeroExtend { src, dst },
        },
    };

    instrs.push(cast_instr);

    dst_var
}

fn postfix_prefix_instrs(
    instrs: &mut Vec<Instr>,
    symbols: &mut Namespace<TypeCtx>,
    op: ast::UnaryOp,
    expr: &ast::TypedExpr,
    expr_type: &Type,
) -> ExprResult {
    let lhs = expr.to_ir(instrs, symbols);
    let (op, cache) = match op {
        ast::UnaryOp::IncPost => (ast::BinaryOp::Add, true),
        ast::UnaryOp::DecPost => (ast::BinaryOp::Subtract, true),

        ast::UnaryOp::IncPre => (ast::BinaryOp::Add, false),
        ast::UnaryOp::DecPre => (ast::BinaryOp::Subtract, false),
        _ => unreachable!(),
    };
    let op = op.to_ir();
    let one = match expr.type_.as_ref().expect("operand type info must be known") {
        Type::Int => Const::Int(1),
        Type::Long => Const::Long(1),
        Type::UInt => Const::UInt(1),
        Type::ULong => Const::ULong(1),
        Type::Double => Const::Double(1.0),
        Type::Func { .. } | Type::Pointer { .. } | Type::Array { .. } => unreachable!(),
    };
    let result = make_ir_variable("inc", expr_type.clone(), symbols);

    match lhs {
        ExprResult::Plain(obj) => {
            let Value::Var(dst) = obj else { unreachable!() };
            if cache {
                instrs.push(Instr::Copy { src: Value::Var(dst.clone()), dst: result.clone() });
            }
            instrs.push(Instr::Binary {
                op,
                lhs: Value::Var(dst.clone()),
                rhs: Value::Const(one),
                dst: dst.clone(),
            });
            ExprResult::Plain(if cache { Value::Var(result) } else { Value::Var(dst) })
        }
        ExprResult::DerefPtr(ptr) => {
            let dst = make_ir_variable("drf", expr_type.clone(), symbols);
            instrs.push(Instr::Load { src_ptr: ptr.clone(), dst: dst.clone() });
            let Value::Var(ptr) = ptr else { unreachable!("pointer cannot be a constant") };

            if cache {
                instrs.push(Instr::Copy { src: Value::Var(dst.clone()), dst: result.clone() });
            }
            instrs.extend([
                Instr::Binary {
                    op,
                    lhs: Value::Var(dst.clone()),
                    rhs: Value::Const(one),
                    dst: dst.clone(),
                },
                Instr::Store { src: Value::Var(dst.clone()), dst_ptr: ptr },
            ]);

            ExprResult::Plain(if cache { Value::Var(result) } else { Value::Var(dst) })
        }
    }
}

fn logical_ops_instrs(
    instrs: &mut Vec<Instr>,
    symbols: &mut Namespace<TypeCtx>,
    op: ast::BinaryOp,
    lhs: &ast::TypedExpr,
    rhs: &ast::TypedExpr,
    expr_type: &Type,
) -> ExprResult {
    let counter = GEN.fetch_add(1, Relaxed);

    let or = matches!(op, ast::BinaryOp::Or);

    let cond_jump = eco_format!("jmp.{}", counter);
    let end = eco_format!("end.{}", counter);

    let dst = make_ir_variable("lgc", expr_type.clone(), symbols);

    let v1 = lhs.to_ir_and_convert(instrs, symbols);
    instrs.push(if or {
        Instr::JumpIfNotZero { cond: v1, target: cond_jump.clone() }
    } else {
        Instr::JumpIfZero { cond: v1, target: cond_jump.clone() }
    });

    let v2 = rhs.to_ir_and_convert(instrs, symbols);
    instrs.extend([
        if or {
            Instr::JumpIfNotZero { cond: v2, target: cond_jump.clone() }
        } else {
            Instr::JumpIfZero { cond: v2, target: cond_jump.clone() }
        },
        Instr::Copy { src: Value::Const(Const::Int(i32::from(!or))), dst: dst.clone() },
        Instr::Jump { target: end.clone() },
        Instr::Label(cond_jump),
        Instr::Copy { src: Value::Const(Const::Int(i32::from(or))), dst: dst.clone() },
        Instr::Label(end),
    ]);

    ExprResult::Plain(Value::Var(dst))
}

impl ast::UnaryOp {
    fn to_ir(self) -> UnOp {
        match self {
            Self::Complement => UnOp::Complement,
            Self::Negate => UnOp::Negate,
            Self::Not => UnOp::Not,

            Self::Plus
            | Self::IncPre
            | Self::IncPost
            | Self::DecPre
            | Self::DecPost
            | Self::AddressOf
            | Self::Dereference => unreachable!("implemented in expr.to_ir"),
        }
    }
}
impl ast::BinaryOp {
    fn to_ir(self) -> BinOp {
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
impl ast::Block {
    fn to_ir(&self, instrs: &mut Vec<Instr>, symbols: &mut Namespace<TypeCtx>) {
        for bi in &self.0 {
            bi.to_ir(instrs, symbols);
        }
    }
}
impl ast::BlockItem {
    fn to_ir(&self, instrs: &mut Vec<Instr>, symbols: &mut Namespace<TypeCtx>) {
        match self {
            Self::S(stmt) => {
                stmt.to_ir(instrs, symbols);
            }
            Self::D(decl) => {
                decl.to_ir(instrs, symbols);
            }
        }
    }
}
impl ast::Decl {
    fn to_ir(&self, instrs: &mut Vec<Instr>, symbols: &mut Namespace<TypeCtx>) {
        match self {
            Self::Func(func @ ast::FuncDecl { body: Some(_), .. }) => {
                func.to_ir(instrs, symbols);
            }
            Self::Func(_) => {}
            Self::Var(var) => var.to_ir(instrs, symbols),
        };
    }
}
impl ast::VarDecl {
    fn to_ir(&self, instrs: &mut Vec<Instr>, symbols: &mut Namespace<TypeCtx>) {
        match &self.init {
            Some(ast::Initializer::Single(e)) => {
                let Some(TypeCtx { type_: dst_type, .. }) = symbols.get(&self.name) else {
                    unreachable!()
                };
                let v = if *dst_type == e.type_.clone().unwrap() {
                    e.to_ir_and_convert(instrs, symbols)
                } else {
                    ast::Expr::Cast { target: dst_type.clone(), inner: Box::new(e.clone()) }
                        .typed(dst_type.clone())
                        .to_ir_and_convert(instrs, symbols)
                };

                instrs.push(Instr::Copy { src: v, dst: Place(self.name.clone()) });
            }
            Some(ast::Initializer::Compound(inits)) => {
                let exprs = inits.clone().into_iter().flat_map(|i| i.flatten_exprs());

                let mut offset = 0;
                for expr in exprs {
                    let ir_expr = expr.to_ir_and_convert(instrs, symbols);
                    instrs.push(Instr::CopyToOffset {
                        src: ir_expr,
                        dst: self.name.clone(),
                        offset,
                    });
                    offset += expr.type_.unwrap().size();
                }
            }
            None => {}
        }
    }
}

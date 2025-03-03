use crate::{assembly, ast};
use ecow::{EcoString, eco_format};
use either::Either::{Left, Right};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

#[derive(Debug, Clone)]
pub struct Program(pub Vec<FuncDef>);

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: EcoString,
    pub params: Vec<EcoString>,
    pub body: Vec<Instr>,
}

#[derive(Debug, Clone)]
pub enum Instr {
    Return(Value),
    Unary { op: UnOp, src: Value, dst: Place },
    Binary { op: BinOp, lhs: Value, rhs: Value, dst: Place },
    Copy { src: Value, dst: Place },
    Jump { target: EcoString },
    JumpIfZero { cond: Value, target: EcoString },
    JumpIfNotZero { cond: Value, target: EcoString },
    Label(EcoString),
    FuncCall { name: EcoString, args: Vec<Value>, dst: Place },
}

#[derive(Debug, Clone)]
pub enum Value {
    Const(i32),
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
        Program(self.0.iter().filter_map(|fd| fd.body.as_ref().map(|_| fd.to_ir(instrs))).collect())
    }
}
impl ToIr for ast::FuncDecl {
    type Output = FuncDef;
    fn to_ir(&self, instrs: &mut Vec<Instr>) -> Self::Output {
        let Some(ref body) = self.body else { unreachable!() };
        body.to_ir(instrs);

        if !matches!(instrs.last(), Some(Instr::Return(_))) {
            instrs.push(Instr::Return(Value::Const(0)));
        }

        FuncDef {
            name: self.name.clone(),
            params: self.params.clone(),
            body: std::mem::take(instrs),
        }
    }
}
impl ToIr for ast::Stmt {
    type Output = ();
    #[expect(clippy::too_many_lines)]
    fn to_ir(&self, instrs: &mut Vec<Instr>) -> Self::Output {
        let brk_label = |s| eco_format!("brk.{s}");
        let cntn_label = |s| eco_format!("cntn.{s}");

        let case_label = |s, i| eco_format!("case.{i}.{s}");
        let dfl_label = |s| eco_format!("dflt.{s}");

        match self {
            Self::Return(expr) => {
                let dst = expr.to_ir(instrs);
                instrs.push(Instr::Return(dst));
            }
            Self::Expression(expr) => {
                expr.to_ir(instrs);
            }
            Self::If { cond, then, else_ } => {
                static IF: AtomicUsize = AtomicUsize::new(0);
                let counter = IF.fetch_add(1, Relaxed);

                let else_label = eco_format!("if.else.{}", counter);

                let cond = cond.to_ir(instrs);
                // Do I need Copy Instr here ?
                instrs.push(Instr::JumpIfZero { cond, target: else_label.clone() });

                then.to_ir(instrs);

                if let Some(else_) = else_ {
                    let end_label = eco_format!("if.end.{}", counter);

                    instrs.extend([
                        Instr::Jump { target: end_label.clone() },
                        Instr::Label(else_label),
                    ]);

                    else_.to_ir(instrs);

                    instrs.push(Instr::Label(end_label));
                } else {
                    instrs.push(Instr::Label(else_label));
                }
            }

            Self::Compound(b) => b.to_ir(instrs),

            Self::GoTo(label) => {
                instrs.push(Instr::Jump { target: eco_format!(".Lgoto..{label}") });
            }
            Self::Label(label, stmt) => {
                instrs.push(Instr::Label(eco_format!(".Lgoto..{label}")));
                stmt.to_ir(instrs);
            }

            Self::Null => {}

            Self::Switch { ctrl, body, label: Some(label), cases } => {
                static SWTCH: AtomicUsize = AtomicUsize::new(0);

                let dst = Place(eco_format!("swch.tmp.{}", SWTCH.fetch_add(1, Relaxed)));

                let ctrl = ctrl.to_ir(instrs);

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

                body.to_ir(instrs);
                instrs.push(Instr::Label(brk_label(label)));
            }
            Self::Case { cnst, body, label: Some(label) } => {
                let ast::Expr::ConstInt(value) = cnst else { unreachable!() };
                instrs.push(Instr::Label(case_label(label, *value)));

                body.to_ir(instrs);
            }
            Self::Default { body, label: Some(label) } => {
                instrs.push(Instr::Label(dfl_label(label)));

                body.to_ir(instrs);
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
                body.to_ir(instrs);

                instrs.push(Instr::Label(cntn_label(label)));
                let cond = cond.to_ir(instrs);

                instrs.extend([
                    Instr::JumpIfNotZero { cond, target: start_label },
                    Instr::Label(brk_label(label)),
                ]);
            }
            Self::While { cond, body, label: Some(label) } => {
                let start_label = cntn_label(label);
                let end_label = brk_label(label);
                instrs.push(Instr::Label(start_label.clone()));

                let cond = cond.to_ir(instrs);
                instrs.push(Instr::JumpIfZero { cond, target: end_label.clone() });
                body.to_ir(instrs);

                instrs.extend([Instr::Jump { target: start_label }, Instr::Label(end_label)]);
            }
            Self::For { init, cond, post, body, label: Some(label) } => {
                let start_label = eco_format!("strt.{label}");
                let cntn_label = cntn_label(label);
                let brk_label = brk_label(label);

                match init {
                    Left(decl) => decl.to_ir(instrs),
                    Right(Some(expr)) => drop(expr.to_ir(instrs)),
                    Right(None) => (),
                }

                instrs.push(Instr::Label(start_label.clone()));

                if let Some(cond) = cond {
                    let cond = cond.to_ir(instrs);
                    instrs.push(Instr::JumpIfZero { cond, target: brk_label.clone() });
                }

                body.to_ir(instrs);

                instrs.push(Instr::Label(cntn_label.clone()));
                if let Some(post) = post {
                    post.to_ir(instrs);
                }

                instrs.extend([Instr::Jump { target: start_label }, Instr::Label(brk_label)]);
            }
            any => unreachable!("statement shouldn't exist: {any:?}"),
        }
    }
}
impl ToIr for ast::Expr {
    type Output = Value;
    fn to_ir(&self, instrs: &mut Vec<Instr>) -> Self::Output {
        match self {
            Self::ConstInt(i) => Value::Const(*i),
            Self::Unary(ast::UnaryOp::Plus, expr) => expr.to_ir(instrs),
            Self::Unary(
                op @ (ast::UnaryOp::IncPost
                | ast::UnaryOp::DecPost
                | ast::UnaryOp::IncPre
                | ast::UnaryOp::DecPre),
                expr,
            ) => postfix_prefix_instrs(instrs, *op, expr),
            Self::Unary(unary_op, expr) => {
                static UNARY_TMP: AtomicUsize = AtomicUsize::new(0);

                let src = expr.to_ir(instrs);

                let dst_name = eco_format!("unop.tmp.{}", UNARY_TMP.fetch_add(1, Relaxed));
                let dst = Place(dst_name);
                let op = unary_op.to_ir(instrs);

                instrs.push(Instr::Unary { op, src, dst: dst.clone() });

                Value::Var(dst)
            }
            Self::Binary { op: op @ (ast::BinaryOp::And | ast::BinaryOp::Or), lhs, rhs } => {
                logical_ops_instrs(instrs, *op, lhs, rhs)
            }
            Self::Binary { op, lhs, rhs } => {
                static BINARY_TMP: AtomicUsize = AtomicUsize::new(0);

                let lhs = lhs.to_ir(instrs);
                let rhs = rhs.to_ir(instrs);

                let dst_name = eco_format!("binop.tmp.{}", BINARY_TMP.fetch_add(1, Relaxed));
                let dst = Place(dst_name);

                let op = op.to_ir(instrs);

                instrs.push(Instr::Binary { op, lhs, rhs, dst: dst.clone() });
                Value::Var(dst)
            }
            Self::Var(id) => Value::Var(Place(id.clone())),
            Self::Assignemnt(place, value) => {
                let ast::Expr::Var(ref dst) = **place else {
                    unreachable!("place expression should be resolved earlier.")
                };

                let rhs = value.to_ir(instrs);

                instrs.push(Instr::Copy { src: rhs, dst: Place(dst.clone()) });

                Value::Var(Place(dst.clone()))
            }
            Self::CompoundAssignment { op, lhs, rhs } => {
                let ast::Expr::Var(ref dst) = **lhs else {
                    unreachable!("place expression should be resolved earlier.")
                };
                let ret =
                    Self::Binary { op: *op, lhs: lhs.clone(), rhs: rhs.clone() }.to_ir(instrs);

                instrs.push(Instr::Copy { src: ret, dst: Place(dst.clone()) });

                Value::Var(Place(dst.clone()))
            }

            Self::Conditional { cond, then, else_ } => {
                static TERNARY: AtomicUsize = AtomicUsize::new(0);
                let counter = TERNARY.fetch_add(1, Relaxed);

                let end_label = eco_format!("ter.end.{}", counter);
                let e2_label = eco_format!("ter.e2.{}", counter);
                let result = Place(eco_format!("ter.tmp.{}", counter));

                let cond = cond.to_ir(instrs);
                instrs.push(Instr::JumpIfZero { cond, target: e2_label.clone() });

                let v1 = then.to_ir(instrs);
                instrs.extend([
                    Instr::Copy { src: v1, dst: result.clone() },
                    Instr::Jump { target: end_label.clone() },
                    Instr::Label(e2_label),
                ]);

                let v2 = else_.to_ir(instrs);
                instrs.extend([
                    Instr::Copy { src: v2, dst: result.clone() },
                    Instr::Label(end_label),
                ]);

                Value::Var(result)
            }

            Self::FuncCall { name, args } => {
                static COUNTER: AtomicUsize = AtomicUsize::new(0);
                let dst = eco_format!("fcall.tmp.{}", COUNTER.fetch_add(1, Relaxed));
                let dst = Place(dst);

                let vals = args.iter().map(|arg| arg.to_ir(instrs)).collect();

                instrs.push(Instr::FuncCall { name: name.clone(), args: vals, dst: dst.clone() });

                Value::Var(dst)
            }
        }
    }
}

fn postfix_prefix_instrs(instrs: &mut Vec<Instr>, op: ast::UnaryOp, expr: &ast::Expr) -> Value {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    let ast::Expr::Var(ref dst) = *expr else {
        unreachable!("place expression should be resolved earlier.")
    };

    let (op, cache) = match op {
        ast::UnaryOp::IncPost => (ast::BinaryOp::Add, true),
        ast::UnaryOp::DecPost => (ast::BinaryOp::Subtract, true),

        ast::UnaryOp::IncPre => (ast::BinaryOp::Add, false),
        ast::UnaryOp::DecPre => (ast::BinaryOp::Subtract, false),
        _ => unreachable!(),
    };

    let tmp = eco_format!("fix.tmp.{}", COUNTER.fetch_add(1, Relaxed));
    let tmp = Place(tmp);

    let var = Place(dst.clone());

    if cache {
        instrs.push(Instr::Copy { src: Value::Var(var.clone()), dst: tmp.clone() });
    }

    let op = op.to_ir(instrs);

    instrs.push(Instr::Binary {
        op,
        lhs: Value::Var(var.clone()),
        rhs: Value::Const(1),
        dst: var.clone(),
    });

    if cache { Value::Var(tmp) } else { Value::Var(var) }
}

fn logical_ops_instrs(
    instrs: &mut Vec<Instr>,
    op: ast::BinaryOp,
    lhs: &ast::Expr,
    rhs: &ast::Expr,
) -> Value {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let counter = COUNTER.fetch_add(1, Relaxed);

    let or = matches!(op, ast::BinaryOp::Or);

    let cond_jump = eco_format!("lgc.jmp.{}", counter);
    let end = eco_format!("lgc.end.{}", counter);

    let dst_name = eco_format!("lgc.tmp.{}", counter);
    let dst = Place(dst_name);

    let v1 = lhs.to_ir(instrs);
    instrs.push(if or {
        Instr::JumpIfNotZero { cond: v1, target: cond_jump.clone() }
    } else {
        Instr::JumpIfZero { cond: v1, target: cond_jump.clone() }
    });

    let v2 = rhs.to_ir(instrs);
    instrs.extend([
        if or {
            Instr::JumpIfNotZero { cond: v2, target: cond_jump.clone() }
        } else {
            Instr::JumpIfZero { cond: v2, target: cond_jump.clone() }
        },
        Instr::Copy { src: Value::Const(i32::from(!or)), dst: dst.clone() },
        Instr::Jump { target: end.clone() },
        Instr::Label(cond_jump),
        Instr::Copy { src: Value::Const(i32::from(or)), dst: dst.clone() },
        Instr::Label(end),
    ]);

    Value::Var(dst)
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
impl ToIr for ast::Block {
    type Output = ();

    fn to_ir(&self, instrs: &mut Vec<Instr>) -> Self::Output {
        for bi in &self.0 {
            bi.to_ir(instrs);
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
        match self {
            Self::Func(func @ ast::FuncDecl { body: Some(_), .. }) => {
                func.to_ir(instrs);
            }
            Self::Func(_) => {}
            Self::Var(var) => var.to_ir(instrs),
        };
    }
}
impl ToIr for ast::VarDecl {
    type Output = ();

    fn to_ir(&self, instrs: &mut Vec<Instr>) -> Self::Output {
        if let Some(e) = &self.init {
            let v = e.to_ir(instrs);
            instrs.push(Instr::Copy { src: v, dst: Place(self.name.clone()) });
        }
    }
}

use ecow::EcoString;

pub trait Node {}

#[derive(Debug)]
pub struct Program(pub FuncDef);
impl Node for Program {}

#[derive(Debug)]
pub struct FuncDef {
    pub name: Ident,
    pub body: Stmt,
}
impl Node for FuncDef {}

#[derive(Debug)]
pub struct Ident(pub EcoString);
impl Node for Ident {}

#[derive(Debug)]
pub enum Stmt {
    Return(Box<Expr>),
    If {
        cond: Box<Expr>,
        then: Box<Stmt>,
        else_: Option<Box<Stmt>>,
    },
}
impl Node for Stmt {}

#[derive(Debug)]
pub enum Expr {
    ConstInt(usize),
}
impl Node for Expr {}

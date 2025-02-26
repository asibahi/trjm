use ecow::EcoString;

pub trait Tac : std::fmt::Debug + Clone{}
impl<T> Tac for Vec<T> where T: Tac {}

#[derive(Debug, Clone)]
pub struct Program(pub FuncDef);
impl Tac for Program {}

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: EcoString,
    pub body: Vec<Instr>,
}
impl Tac for FuncDef {}

#[derive(Debug, Clone)]
pub enum Instr {
    Return(Value),
    Unary { op: UnOp, src: Value, dst: Place },
}
impl Tac for Instr {}

#[derive(Debug, Clone)]
pub enum Value {
    Const(u32),
    Var(Place),
}
impl Tac for Value {}

#[derive(Debug, Clone)]
pub struct Place(pub EcoString);
impl Tac for Place {}

#[derive(Debug, Clone)]
pub enum UnOp {
    Complement,
    Negate,
}
impl Tac for UnOp {}

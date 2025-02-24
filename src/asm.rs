use ecow::EcoString;

pub trait Asm {}

#[derive(Debug)]
pub struct Program(pub FuncDef);
impl Asm for Program {}

#[derive(Debug)]
pub struct FuncDef {
    pub name: Ident,
    pub instrs: Vec<Instr>,
}
impl Asm for FuncDef {}

#[derive(Debug)]
pub enum Instr {
    Mov { src: Operand, dst: Operand },
    Ret,
}
impl Asm for Instr {}
impl Asm for Vec<Instr> {}

#[derive(Debug)]
pub enum Operand {
    Imm(usize),
    Register,
}
impl Asm for Operand {}

#[derive(Debug)]
pub struct Ident(pub EcoString);
impl Asm for Ident {}

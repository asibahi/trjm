use ecow::EcoString;
use std::io::Write;

pub trait Asm {
    fn emit_code(&self, f: &mut impl Write);
}

impl<T> Asm for Vec<T>
where
    T: Asm,
{
    fn emit_code(&self, f: &mut impl Write) {
        for elem in self {
            elem.emit_code(f);
        }
    }
}

#[derive(Debug)]
pub struct Program(pub FuncDef);
impl Asm for Program {
    fn emit_code(&self, f: &mut impl Write) {
        self.0.emit_code(f);
    }
}

#[derive(Debug)]
pub struct FuncDef {
    pub name: EcoString,
    pub instrs: Vec<Instr>,
}
impl Asm for FuncDef {
    fn emit_code(&self, f: &mut impl Write) {
        _ = writeln!(f, "\t.globl _{}", self.name);
        _ = writeln!(f, "_{}:", self.name);
        self.instrs.emit_code(f);
    }
}

#[derive(Debug)]
pub enum Instr {
    Mov { src: Operand, dst: Operand },
    Ret,
}
impl Asm for Instr {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Instr::Mov { src, dst } => {
                _ = write!(f, "\tmovl ");
                src.emit_code(f);
                _ = write!(f, ", ");
                dst.emit_code(f);
                _ = writeln!(f);
            }
            Instr::Ret => {
                _ = writeln!(f, "\tret");
            }
        }
    }
}

#[derive(Debug)]
pub enum Operand {
    Imm(u32),
    Register,
}
impl Asm for Operand {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Operand::Imm(i) => {
                _ = write!(f, "${i}");
            }
            Operand::Register => {
                _ = write!(f, "%eax");
            }
        }
    }
}

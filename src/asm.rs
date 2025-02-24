use ecow::EcoString;
use std::io::Write;

pub trait Asm {
    fn emit_code(&self, f: &mut impl Write);
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
    pub name: Ident,
    pub instrs: Vec<Instr>,
}
impl Asm for FuncDef {
    fn emit_code(&self, f: &mut impl Write) {
        write!(f, "\t.globl _").unwrap();
        self.name.emit_code(f);
        writeln!(f).unwrap();

        write!(f, "_").unwrap();
        self.name.emit_code(f);
        writeln!(f, ":").unwrap();

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
                write!(f, "\tmovl ").unwrap();
                src.emit_code(f);
                write!(f, ", ").unwrap();
                dst.emit_code(f);
                writeln!(f).unwrap();
            }
            Instr::Ret => {
                writeln!(f, "\tret").unwrap();
            }
        }
    }
}
impl Asm for Vec<Instr> {
    fn emit_code(&self, f: &mut impl Write) {
        for instr in self {
            instr.emit_code(f);
        }
    }
}

#[derive(Debug)]
pub enum Operand {
    Imm(usize),
    Register,
}
impl Asm for Operand {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Operand::Imm(i) => {
                write!(f, "${i}").unwrap();
            }
            Operand::Register => {
                write!(f, "%eax").unwrap();
            }
        }
    }
}

#[derive(Debug)]
pub struct Ident(pub EcoString);
impl Asm for Ident {
    fn emit_code(&self, f: &mut impl Write) {
        write!(f, "{}", self.0).unwrap();
    }
}

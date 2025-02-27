use ecow::EcoString;
use rustc_hash::FxHashMap;
use std::{collections::hash_map::Entry, io::Write};

pub trait Assembly {
    fn emit_code(&self, f: &mut impl Write);

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, u32>, stack_depth: &mut u32);
    fn adjust_instrs(&mut self, stack_depth: u32);
}

impl<T> Assembly for Vec<T>
where
    T: Assembly,
{
    fn emit_code(&self, f: &mut impl Write) {
        self.iter().for_each(|e| e.emit_code(f));
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, u32>, stack_depth: &mut u32) {
        self.iter_mut()
            .for_each(|e| e.replace_pseudos(map, stack_depth));
    }

    fn adjust_instrs(&mut self, stack_depth: u32) {
        self.iter_mut().for_each(|e| e.adjust_instrs(stack_depth));
    }
}

#[derive(Debug, Clone)]
pub struct Program(pub FuncDef);
impl Program {
    pub fn fixup_passes(&mut self) {
        let mut map = FxHashMap::<EcoString, u32>::default();
        let mut stack_depth = 0;

        self.replace_pseudos(&mut map, &mut stack_depth);
        self.adjust_instrs(stack_depth);
    }
}
impl Assembly for Program {
    fn emit_code(&self, f: &mut impl Write) {
        self.0.emit_code(f);
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, u32>, stack_depth: &mut u32) {
        self.0.replace_pseudos(map, stack_depth);
    }

    fn adjust_instrs(&mut self, stack_depth: u32) {
        self.0.adjust_instrs(stack_depth);
    }
}

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: EcoString,
    pub instrs: Vec<Instr>,
}
impl Assembly for FuncDef {
    fn emit_code(&self, f: &mut impl Write) {
        _ = writeln!(f, "\t.globl _{}", self.name);
        _ = writeln!(f, "_{}:", self.name);

        _ = writeln!(f, "\tpushq %rbp");
        _ = writeln!(f, "\tmovq %rsp, %rbp");

        self.instrs.emit_code(f);
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, u32>, stack_depth: &mut u32) {
        self.instrs.replace_pseudos(map, stack_depth);
    }

    fn adjust_instrs(&mut self, stack_depth: u32) {
        '_stack_frame: {
            self.instrs.insert(0, Instr::AllocateStack(stack_depth));
        }

        '_validate_movs: {
            let mut out = Vec::with_capacity(self.instrs.len());

            for instr in std::mem::take(&mut self.instrs) {
                match instr {
                    Instr::Mov(src @ Operand::Stack(_), dst @ Operand::Stack(_)) => out.extend([
                        Instr::Mov(src, Operand::Reg(Register::R10)),
                        Instr::Mov(Operand::Reg(Register::R10), dst),
                    ]),
                    Instr::Idiv(v @ Operand::Imm(_)) => out.extend([
                        Instr::Mov(v, Operand::Reg(Register::R10)),
                        Instr::Idiv(Operand::Reg(Register::R10)),
                    ]),
                    Instr::Binary(
                        opp @ (Operator::Add | Operator::Sub),
                        src @ Operand::Stack(_),
                        dst @ Operand::Stack(_),
                    ) => out.extend([
                        Instr::Mov(src, Operand::Reg(Register::R10)),
                        Instr::Binary(opp, Operand::Reg(Register::R10), dst),
                    ]),
                    Instr::Binary(Operator::Mul, src, dst @ Operand::Stack(_)) => out.extend([
                        Instr::Mov(dst.clone(), Operand::Reg(Register::R11)),
                        Instr::Binary(Operator::Mul, src, Operand::Reg(Register::R11)),
                        Instr::Mov(Operand::Reg(Register::R11), dst),
                    ]),
                    other => out.push(other),
                }
            }
            self.instrs = out;
        }
    }
}

#[derive(Debug, Clone)]
pub enum Instr {
    Mov(Operand, Operand),
    Unary(Operator, Operand),
    Binary(Operator, Operand, Operand),
    Idiv(Operand),
    Cdq,
    AllocateStack(u32),
    Ret,
}
impl Assembly for Instr {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Self::Mov(src, dst) => {
                _ = write!(f, "\tmovl ");
                src.emit_code(f);
                _ = write!(f, ", ");
                dst.emit_code(f);
                _ = writeln!(f);
            }
            Instr::Unary(un_op, operand) => {
                _ = write!(f, "\t");
                un_op.emit_code(f);
                _ = write!(f, " ");
                operand.emit_code(f);
                _ = writeln!(f);
            }
            Instr::Binary(bin_op, op1, op2) => {
                _ = write!(f, "\t");
                bin_op.emit_code(f);
                _ = write!(f, " ");
                op1.emit_code(f);
                _ = write!(f, ", ");
                op2.emit_code(f);
                _ = writeln!(f);
            }
            Instr::Idiv(operand) => {
                _ = write!(f, "\tidivl ");
                operand.emit_code(f);
                _ = writeln!(f);
            }
            Instr::Cdq => _ = writeln!(f, "\tcdq"),
            Instr::AllocateStack(i) => _ = writeln!(f, "\tsubq ${i}, %rsp"),
            Self::Ret => {
                _ = writeln!(f, "\tmovq %rbp, %rsp");
                _ = writeln!(f, "\tpopq %rbp");
                _ = writeln!(f, "\tret");
            }
        }
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, u32>, stack_depth: &mut u32) {
        match self {
            Instr::Mov(src, dst) => {
                src.replace_pseudos(map, stack_depth);
                dst.replace_pseudos(map, stack_depth);
            }
            Instr::Unary(_, opp) | Instr::Idiv(opp) => {
                opp.replace_pseudos(map, stack_depth);
            }
            Instr::Binary(bin_op, opp1, opp2) => {
                bin_op.replace_pseudos(map, stack_depth);
                opp1.replace_pseudos(map, stack_depth);
                opp2.replace_pseudos(map, stack_depth);
            }
            Instr::Cdq | Instr::AllocateStack(_) | Instr::Ret => (),
        }
    }

    fn adjust_instrs(&mut self, _: u32) {}
}

#[derive(Debug, Clone)]
pub enum Operand {
    Imm(u32),
    Reg(Register),
    Pseudo(EcoString),
    Stack(u32),
}
impl Assembly for Operand {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Self::Imm(i) => _ = write!(f, "${i}"),
            Self::Reg(r) => r.emit_code(f),
            Self::Stack(i) => _ = write!(f, "-{i}(%rbp)"),
            Self::Pseudo(e) => unreachable!("pseudo register {e} printed"),
        }
    }

    fn replace_pseudos(&mut self, map: &mut FxHashMap<EcoString, u32>, stack_depth: &mut u32) {
        let Operand::Pseudo(name) = self else {
            return;
        };

        let stack = if let Entry::Vacant(e) = map.entry(name.clone()) {
            *stack_depth += 4; // 4 bytes
            e.insert(*stack_depth);
            *stack_depth
        } else {
            map[name]
        };

        std::mem::swap(self, &mut Operand::Stack(stack));
    }
    fn adjust_instrs(&mut self, _: u32) {}
}

#[derive(Debug, Clone, Copy)]
pub enum Operator {
    Not,
    Neg,
    Add,
    Sub,
    Mul,
}
impl Assembly for Operator {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Operator::Not => _ = write!(f, "notl"),
            Operator::Neg => _ = write!(f, "negl"),

            Operator::Add => _ = write!(f, "addl"),
            Operator::Sub => _ = write!(f, "subl"),
            Operator::Mul => _ = write!(f, "imull"),
        }
    }

    fn replace_pseudos(&mut self, _: &mut FxHashMap<EcoString, u32>, _: &mut u32) {}
    fn adjust_instrs(&mut self, _: u32) {}
}


#[derive(Debug, Clone, Copy)]
pub enum Register {
    AX,
    DX,
    R10,
    R11,
}
impl Assembly for Register {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Register::AX => _ = write!(f, "%eax"),
            Register::DX => _ = write!(f, "%edx"),

            Register::R10 => _ = write!(f, "%r10d"),
            Register::R11 => _ = write!(f, "%r11d"),
        }
    }

    fn replace_pseudos(&mut self, _: &mut FxHashMap<EcoString, u32>, _: &mut u32) {}
    fn adjust_instrs(&mut self, _: u32) {}
}

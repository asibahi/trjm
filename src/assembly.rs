use crate::{
    ast::{self, Attributes::*, Namespace, StaticInit, Type, TypeCtx},
    ir::{self, GEN},
};
use ecow::{EcoString as Identifier, eco_format};
use either::Either::{self, Left, Right};
use std::{io::Write, sync::atomic::Ordering::Relaxed};

#[derive(Debug, Clone, Copy)]
#[expect(dead_code)]
enum BSymbol {
    Obj { type_: AsmType, signed: bool, is_static: bool, constant: bool },
    Func { _defined: bool },
}
impl BSymbol {
    fn new_obj(type_: AsmType, signed: bool, is_static: bool, constant: bool) -> Self {
        Self::Obj { type_, signed, is_static, constant }
    }
}
impl From<TypeCtx> for BSymbol {
    fn from(value: TypeCtx) -> Self {
        let TypeCtx { type_, attr } = value;

        match (type_, attr) {
            (Type::Func { .. }, Func { defined, .. }) => Self::Func { _defined: defined },
            (_, Func { .. }) | (Type::Func { .. }, _) => unreachable!(),
            (Type::Int, Static { .. }) => Self::new_obj(Longword, true, true, false),
            (Type::UInt, Static { .. }) => Self::new_obj(Longword, false, true, false),
            (Type::Int, Local) => Self::new_obj(Longword, true, false, false),
            (Type::UInt, Local) => Self::new_obj(Longword, false, false, false),
            (Type::Long, Static { .. }) => Self::new_obj(Quadword, true, true, false),
            (Type::ULong | Type::Pointer { .. }, Static { .. }) => {
                Self::new_obj(Quadword, false, true, false)
            }
            (Type::Long, Local) => Self::new_obj(Quadword, true, false, false),
            (Type::ULong | Type::Pointer { .. }, Local) => {
                Self::new_obj(Quadword, false, false, false)
            }
            (Type::Double, Static { .. }) => Self::new_obj(Doubleword, false, true, false),
            (Type::Double, Local) => Self::new_obj(Doubleword, false, false, false),
            (Type::Array { .. }, _) => todo!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Program {
    top_level: Vec<TopLevel>,
    symbols: Namespace<BSymbol>,
}
impl Program {
    pub fn fixup_passes(&mut self) {
        self.replace_pseudos(&self.symbols.clone());
        self.adjust_instrs();
    }

    pub fn emit_code(&self, f: &mut impl Write) {
        self.top_level.iter().for_each(|tl| tl.emit_code(f));
    }

    fn replace_pseudos(&mut self, symbols: &Namespace<BSymbol>) {
        self.top_level.iter_mut().for_each(|tl| tl.replace_pseudos(symbols));
    }

    fn adjust_instrs(&mut self) {
        self.top_level.iter_mut().for_each(TopLevel::adjust_instrs);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsmType {
    Longword,
    Quadword,
    Doubleword, // ?
}
use AsmType::*;

impl AsmType {
    fn width(self) -> u8 {
        match self {
            Longword => 4,
            Quadword | Doubleword => 8,
        }
    }
    fn return_register(self) -> Operand {
        match self {
            Longword => Reg(AX, 4),
            Quadword => Reg(AX, 8),
            Doubleword => Reg(XMM0, 8),
        }
    }

    fn emit_code(self) -> &'static str {
        match self {
            Longword => "l",
            Quadword => "q",
            Doubleword => "sd", // NOT FOR XOR
        }
    }
}

#[derive(Debug, Clone)]
pub enum TopLevel {
    Function { name: Identifier, instrs: Vec<Instr>, stack_size: i32, global: bool },
    StaticVariable { name: Identifier, global: bool, init: StaticInit, alignment: i32 },
    StaticConstant { name: Identifier, init: StaticInit, alignment: i32 },
}

impl TopLevel {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Self::Function { name, instrs, global, .. } => {
                if *global {
                    _ = writeln!(f, "\t.globl  _{name}");
                }
                _ = writeln!(f, "\t.text");
                _ = writeln!(f, "_{name}:");

                _ = writeln!(f, "\tpushq   %rbp");
                _ = writeln!(f, "\tmovq    %rsp,   %rbp");

                instrs.iter().for_each(|i| i.emit_code(f));
            }
            Self::StaticVariable { name, global, init, alignment } if init.is_zero() => {
                if *global {
                    _ = writeln!(f, "\t.globl  _{name}");
                }
                _ = writeln!(f, "\t.bss");
                _ = writeln!(f, "\t.balign {alignment}");
                _ = writeln!(f, "_{name}:");
                _ = writeln!(f, "\t.zero   {alignment}");
            }
            Self::StaticVariable { name, global, init, alignment } => {
                if *global {
                    _ = writeln!(f, "\t.globl  _{name}");
                }
                _ = writeln!(f, "\t.data");
                _ = writeln!(f, "\t.balign {alignment}");
                _ = writeln!(f, "_{name}:");
                _ = writeln!(f, "\t.{init}");
            }
            Self::StaticConstant { name, init, alignment } => {
                match alignment {
                    8 => _ = writeln!(f, "\t.literal8"),
                    16 => _ = writeln!(f, "\t.literal16"),
                    _ => unreachable!(),
                }

                _ = writeln!(f, "\t.balign {alignment}");
                _ = writeln!(f, "_{name}:");
                _ = writeln!(f, "\t.{init}");

                if *alignment == 16 {
                    _ = writeln!(f, "\t.quad   0");
                }
            }
        }
    }

    fn replace_pseudos(&mut self, symbols: &Namespace<BSymbol>) {
        match self {
            Self::Function { instrs, stack_size, .. } => {
                let mut map = Namespace::default();
                let mut stack_depth = 0;

                instrs
                    .iter_mut()
                    .for_each(|inst| inst.replace_pseudos(&mut map, &mut stack_depth, symbols));

                *stack_size = stack_depth;
            }
            Self::StaticVariable { .. } | Self::StaticConstant { .. } => {}
        }
    }

    fn adjust_instrs(&mut self) {
        '_stack_frame: {
            match self {
                Self::Function { instrs, stack_size, .. } => {
                    let sd = stack_size.unsigned_abs().next_multiple_of(16);
                    let instr = Instr::Binary(Operator::Sub, Quadword, Imm(sd.into()), Reg(SP, 8));

                    instrs.insert(0, instr);
                }
                Self::StaticVariable { .. } | Self::StaticConstant { .. } => {}
            }
        }

        '_fixup_instrs: for _ in 0..3 {
            let Self::Function { instrs, .. } = self else {
                break;
            };
            fixup_instruction(instrs);
        }
    }
}

fn fixup_instruction(instrs: &mut Vec<Instr>) {
    let mut out = Vec::with_capacity(instrs.len());

    for instr in std::mem::take(instrs) {
        match instr {
            Instr::Push(op @ Reg(r, _)) if r.is_xmm() => out.extend([
                Instr::Binary(Operator::Sub, Quadword, Imm(8), Reg(SP, 8)),
                Instr::Mov(Doubleword, op, Reg(SP, 8)),
            ]),
            Instr::Lea(src, dst @ (Memory(..) | Data(_))) => {
                out.extend([Instr::Lea(src, Reg(R11, 8)), Instr::Mov(Quadword, Reg(R11, 8), dst)])
            }
            Instr::Mov(ty, Imm(0), dst @ Reg(..)) => {
                out.push(Instr::Binary(Operator::Xor, ty, dst.clone(), dst))
            }
            Instr::Mov(ty, src @ (Memory(..) | Data(_)), dst @ (Memory(..) | Data(_))) => {
                let reg = if ty == Doubleword { XMM15 } else { R10 };
                out.extend([
                    Instr::Mov(ty, src, Reg(reg, ty.width())),
                    Instr::Mov(ty, Reg(reg, ty.width()), dst),
                ]);
            }
            Instr::Mov(ty, src @ Imm(i), dst @ (Memory(..) | Data(_)))
                if i32::try_from(i).is_err() =>
            {
                let reg = if ty == Doubleword { XMM15 } else { R10 };
                out.extend([
                    Instr::Mov(ty, src, Reg(reg, ty.width())),
                    Instr::Mov(ty, Reg(reg, ty.width()), dst),
                ]);
            }
            Instr::Mov(Longword, Imm(i), dst) if i32::try_from(i).is_err() => {
                out.push(Instr::Mov(Longword, Imm(i64::from(i as i32)), dst))
            }
            Instr::Cmp(Doubleword, src, dst @ (Memory(..) | Data(_))) => out.extend([
                Instr::Mov(Doubleword, dst, Reg(XMM15, 8)),
                Instr::Cmp(Doubleword, src, Reg(XMM15, 8)),
            ]),
            Instr::Cmp(ty, src @ Imm(i), dst) if i32::try_from(i).is_err() => out.extend([
                Instr::Mov(ty, src, Reg(R10, ty.width())),
                Instr::Cmp(ty, Reg(R10, ty.width()), dst),
            ]),
            Instr::Cmp(ty, src @ (Memory(..) | Data(_)), dst @ (Memory(..) | Data(_))) => out
                .extend([
                    Instr::Mov(ty, src, Reg(R10, ty.width())),
                    Instr::Cmp(ty, Reg(R10, ty.width()), dst),
                ]),
            Instr::Cmp(ty, src, dst @ Imm(_)) => out.extend([
                Instr::Mov(ty, dst, Reg(R11, ty.width())),
                Instr::Cmp(ty, src, Reg(R11, ty.width())),
            ]),
            Instr::Push(src @ Imm(i)) if i32::try_from(i).is_err() => {
                out.extend([Instr::Mov(Quadword, src, Reg(R10, 8)), Instr::Push(Reg(R10, 8))])
            }
            Instr::Div(ty, v @ Imm(_)) => out.extend([
                Instr::Mov(ty, v, Reg(R10, ty.width())),
                Instr::Div(ty, Reg(R10, ty.width())),
            ]),
            Instr::Idiv(ty, v @ Imm(_)) => out.extend([
                Instr::Mov(ty, v, Reg(R10, ty.width())),
                Instr::Idiv(ty, Reg(R10, ty.width())),
            ]),
            Instr::Binary(
                opp @ (Operator::Add
                | Operator::Sub
                | Operator::Mul
                | Operator::And
                | Operator::Or
                | Operator::Xor
                | Operator::Sar
                | Operator::Shr
                | Operator::Shl),
                ty,
                src @ Imm(i),
                dst,
            ) if i32::try_from(i).is_err() => out.extend([
                Instr::Mov(ty, src, Reg(R10, ty.width())),
                Instr::Binary(opp, ty, Reg(R10, ty.width()), dst),
            ]),
            Instr::Binary(
                opp @ (Operator::Add
                | Operator::Sub
                | Operator::DivDouble
                | Operator::And
                | Operator::Or
                | Operator::Xor),
                ty,
                src @ (Memory(..) | Data(_)),
                dst @ (Memory(..) | Data(_)),
            ) => {
                let reg = if ty == Doubleword { XMM15 } else { R10 };
                let re2 = if ty == Doubleword { XMM14 } else { R11 };
                out.extend([
                    Instr::Mov(ty, src, Reg(reg, ty.width())),
                    Instr::Mov(ty, dst.clone(), Reg(re2, ty.width())),
                    Instr::Binary(opp, ty, Reg(reg, ty.width()), Reg(re2, ty.width())),
                    Instr::Mov(ty, Reg(re2, ty.width()), dst),
                ]);
            }
            Instr::Binary(Operator::Mul, ty, src, dst @ (Memory(..) | Data(_))) => {
                let reg = if ty == Doubleword { XMM14 } else { R11 };
                out.extend([
                    Instr::Mov(ty, dst.clone(), Reg(reg, ty.width())),
                    Instr::Binary(Operator::Mul, ty, src, Reg(reg, ty.width())),
                    Instr::Mov(ty, Reg(reg, ty.width()), dst),
                ]);
            }
            Instr::Binary(
                opp @ (Operator::Shl | Operator::Sar | Operator::Shr),
                ty,
                src,
                dst @ (Memory(..) | Data(_)),
            ) if src != Reg(CX, 1) => out.extend([
                Instr::Mov(ty, src, Reg(CX, ty.width())),
                Instr::Binary(opp, ty, Reg(CX, 1), dst),
            ]),
            Instr::Movsx(src @ Imm(_), dst) => {
                out.extend([Instr::Mov(Longword, src, Reg(R10, 4)), Instr::Movsx(Reg(R10, 4), dst)])
            }
            Instr::Movsx(src, dst @ (Memory(..) | Data(_))) => {
                out.extend([Instr::Movsx(src, Reg(R11, 8)), Instr::Mov(Quadword, Reg(R11, 8), dst)])
            }
            Instr::MovZeroExtend(src, dst @ Reg(_, _)) => {
                out.push(Instr::Mov(Longword, src, dst.align_width(Longword)))
            }
            Instr::MovZeroExtend(src, dst @ (Memory(..) | Data(_))) => out.extend([
                Instr::Mov(Longword, src, Reg(R11, 4)),
                Instr::Mov(Quadword, Reg(R11, 8), dst),
            ]),
            Instr::Cvttsd2si(ty, src, dst @ (Memory(..) | Data(_))) => out.extend([
                Instr::Cvttsd2si(ty, src, Reg(R11, ty.width())),
                Instr::Mov(ty, Reg(R11, ty.width()), dst),
            ]),
            Instr::Cvtsi2sd(ty, src @ Imm(_), dst) => out.extend([
                Instr::Mov(ty, src, Reg(R10, ty.width())),
                Instr::Cvtsi2sd(ty, Reg(R10, ty.width()), dst),
            ]),
            Instr::Cvtsi2sd(ty, src, dst @ (Memory(..) | Data(_))) => out.extend([
                Instr::Cvtsi2sd(ty, src, Reg(XMM15, 8)),
                Instr::Mov(Doubleword, Reg(XMM15, 8), dst),
            ]),

            other => out.push(other),
        }
    }

    *instrs = out;
}

#[derive(Debug, Clone)]
pub enum Instr {
    Mov(AsmType, Operand, Operand),
    Movsx(Operand, Operand),
    MovZeroExtend(Operand, Operand),
    // Load Effective Address
    Lea(Operand, Operand),
    // convert truncate scalar double to signed integer
    Cvttsd2si(AsmType, Operand, Operand),
    Cvtsi2sd(AsmType, Operand, Operand),
    // operations
    Unary(Operator, AsmType, Operand),
    Binary(Operator, AsmType, Operand, Operand),
    Cmp(AsmType, Operand, Operand),
    Idiv(AsmType, Operand), // signed div
    Div(AsmType, Operand),  // unsigned div
    Cdq(AsmType),
    Jmp(Identifier),
    JmpCC(CondCode, Identifier),
    SetCC(CondCode, Operand),
    Label(Identifier),
    Push(Operand),
    Call(Identifier),
    Ret,
}
impl Instr {
    fn emit_code(&self, f: &mut impl Write) {
        match self {
            Self::Mov(ty, src, dst) => {
                let ty = ty.emit_code();
                let name = eco_format!("mov{ty}");

                let src = src.emit_code() + ",";
                let dst = dst.emit_code();

                _ = writeln!(f, "\t{name:<7} {src:<7} {dst}");
            }
            Self::Unary(un_op, ty, operand) => {
                let ty = ty.emit_code();
                let uo = un_op.emit_code();
                let uo = eco_format!("{uo}{ty}");
                let op = operand.emit_code();

                _ = writeln!(f, "\t{uo:<7} {op}");
            }
            Self::Binary(bin_op, ty, op1, op2) if *bin_op == Operator::Xor && *ty == Doubleword => {
                let name = "xorpd";
                let o1 = op1.emit_code() + ",";
                let o2 = op2.emit_code();

                _ = writeln!(f, "\t{name:<7} {o1:<7} {o2}");
            }
            Self::Binary(bin_op, ty, op1, op2) if *bin_op == Operator::Mul && *ty == Doubleword => {
                let name = "mulsd";

                let o1 = op1.emit_code() + ",";
                let o2 = op2.emit_code();

                _ = writeln!(f, "\t{name:<7} {o1:<7} {o2}");
            }

            Self::Binary(bin_op, ty, op1, op2) => {
                let ty = ty.emit_code();
                let bo = bin_op.emit_code();
                let bo = eco_format!("{bo}{ty}");

                let o1 = op1.emit_code() + ",";
                let o2 = op2.emit_code();

                _ = writeln!(f, "\t{bo:<7} {o1:<7} {o2}");
            }
            Self::Idiv(ty, op) => {
                let ty = ty.emit_code();
                let name = eco_format!("idiv{ty}");
                _ = writeln!(f, "\t{name:<7} {}", op.emit_code());
            }
            Self::Div(ty, op) => {
                let ty = ty.emit_code();
                let name = eco_format!("div{ty}");

                _ = writeln!(f, "\t{name:<7} {}", op.emit_code());
            }

            Self::Cdq(ty) => match ty {
                Longword => _ = writeln!(f, "\tcdq"),
                Quadword => _ = writeln!(f, "\tcqo"),
                Doubleword => unreachable!("this command isnt used with Floats"),
            },

            Self::Push(op) => _ = writeln!(f, "\tpushq   {}", op.emit_code()),
            Self::Call(label) => _ = writeln!(f, "\tcall    _{label}"),

            Self::Ret => {
                _ = writeln!(f, "\tmovq    %rbp,   %rsp");
                _ = writeln!(f, "\tpopq    %rbp");
                _ = writeln!(f, "\tret");
            }
            Self::Cmp(ty, op1, op2) => {
                let name = if *ty == Doubleword { "comi" } else { "cmp" };
                let ty = ty.emit_code();
                let name = eco_format!("{name}{ty}");

                let op1 = op1.emit_code() + ",";
                let op2 = op2.emit_code();

                _ = writeln!(f, "\t{name:<7} {op1:<7} {op2}");
            }
            Self::SetCC(cond, op) => {
                let cond = cond.emit_code();
                let name = eco_format!("set{cond}");
                let op = op.emit_code();

                _ = writeln!(f, "\t{name:<7} {op}");
            }
            Self::Jmp(label) => _ = writeln!(f, "\tjmp     .L{label}"),
            Self::JmpCC(cond, label) => {
                let cond = cond.emit_code();
                let name = eco_format!("j{cond}");

                _ = writeln!(f, "\t{name:<7} .L{label}");
            }
            Self::Label(label) => _ = writeln!(f, "  .L{label}:"),
            Self::Movsx(src, dst) => {
                let src = src.emit_code() + ",";
                let dst = dst.emit_code();

                _ = writeln!(f, "\tmovslq  {src:<7} {dst}");
            }
            Self::MovZeroExtend(..) => unimplemented!(),

            Self::Cvtsi2sd(ty, src, dst) => {
                let ty = ty.emit_code();
                let src = src.emit_code() + ",";
                let dst = dst.emit_code();
                _ = writeln!(f, "\tcvtsi2sd{ty} {src:<6} {dst}");
            }
            Self::Cvttsd2si(ty, src, dst) => {
                let ty = ty.emit_code();
                let src = src.emit_code() + ",";
                let dst = dst.emit_code();
                _ = writeln!(f, "\tcvttsd2si{ty} {src:<5} {dst}");
            }
            Self::Lea(src, dst) => {
                let src = src.emit_code() + ",";
                let dst = dst.emit_code();

                _ = writeln!(f, "\tleaq    {src:<5} {dst}");
            }
        }
    }

    fn replace_pseudos(
        &mut self,
        map: &mut Namespace<i32>,
        stack_depth: &mut i32,
        symbols: &Namespace<BSymbol>,
    ) {
        match self {
            Self::Unary(_, _, opp)
            | Self::Idiv(_, opp)
            | Self::Div(_, opp)
            | Self::SetCC(_, opp) => {
                opp.replace_pseudos(map, stack_depth, symbols);
            }
            Self::Binary(_, _, opp1, opp2)
            | Self::Mov(_, opp1, opp2)
            | Self::Cmp(_, opp1, opp2)
            | Self::Cvtsi2sd(_, opp1, opp2)
            | Self::Cvttsd2si(_, opp1, opp2)
            | Self::Lea(opp1, opp2) => {
                opp1.replace_pseudos(map, stack_depth, symbols);
                opp2.replace_pseudos(map, stack_depth, symbols);
            }
            Self::Push(opp) => opp.replace_pseudos(map, stack_depth, symbols),

            Self::Movsx(op1, op2) | Self::MovZeroExtend(op1, op2) => {
                op1.replace_pseudos(map, stack_depth, symbols);
                op2.replace_pseudos(map, stack_depth, symbols);
            }

            Self::Cdq(_)
            | Self::Ret
            | Self::Jmp(..)
            | Self::JmpCC(..)
            | Self::Label(..)
            | Self::Call(_) => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    Imm(i64),
    Reg(Register, u8),
    Pseudo(Identifier),
    Memory(Register, i32),
    Data(Identifier),
}
use Operand::*;
impl Operand {
    fn align_width(self, ty: AsmType) -> Self {
        match (self, ty) {
            (Reg(r, _), Longword) => Reg(r, 4),
            (Reg(r, _), Quadword) => Reg(r, 8),
            (any, _) => any,
        }
    }

    fn emit_code(&self) -> Identifier {
        match self {
            Imm(i) => eco_format!("${i}"),
            Reg(r, s) => r.emit_code(*s).into(),
            Memory(r, i) => eco_format!("{i}({})", r.emit_code(8)),
            Data(name) => eco_format!("_{name}(%rip)"),

            Pseudo(e) => unreachable!("pseudo register {e} printed"),
        }
    }

    fn replace_pseudos(
        &mut self,
        map: &mut Namespace<i32>,
        stack_depth: &mut i32,
        symbols: &Namespace<BSymbol>,
    ) {
        let Pseudo(name) = self else {
            return;
        };

        let mut new_place = if let Some(stack_depth) = map.get(name) {
            Memory(BP, *stack_depth)
        } else if symbols
            .get(name)
            .is_some_and(|tc| matches!(tc, BSymbol::Obj { is_static: true, .. }))
        {
            Data(name.clone())
        } else {
            let Some(BSymbol::Obj { type_, .. }) = symbols.get(name) else { unreachable!() };

            let curr_depth = *stack_depth;
            match (curr_depth.abs() % 8, type_) {
                (0, Quadword | Doubleword) => *stack_depth -= 8,
                (v, Quadword | Doubleword) => *stack_depth -= 8 + (8 - v),
                (_, Longword) => *stack_depth -= 4,
            }
            // do i need this assertion?
            match type_ {
                Longword => assert!(stack_depth.abs() % 4 == 0),
                Quadword | Doubleword => assert!(stack_depth.abs() % 8 == 0),
            }

            map.insert(name.clone(), *stack_depth);
            Memory(BP, *stack_depth)
        };

        std::mem::swap(self, &mut new_place);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    // unary
    Not,
    Neg,
    // binary
    Add,
    Sub,
    Mul,
    DivDouble,
    And,
    Or,
    Xor,
    Sar,
    Shl,
    Shr,
}
impl Operator {
    fn emit_code(self) -> &'static str {
        match self {
            Self::Not => "not",
            Self::Neg => "neg",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "imul",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Sar => "sar",
            Self::Shl => "shl",

            Self::Shr => "shr",
            Self::DivDouble => "div",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Register {
    AX,
    DX,
    DI,
    SI,
    R8,
    R9,
    R10,
    R11,
    CX,
    SP, // Stack Pointer
    BP, // Base Pointer

    XMM0,
    XMM1,
    XMM2,
    XMM3,
    XMM4,
    XMM5,
    XMM6,
    XMM7,
    XMM14,
    XMM15,
}
const ARG_REGISTERS: [Register; 6] = [DI, SI, DX, CX, R8, R9];
const DOUBLE_ARG_REGISTER: [Register; 8] = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7];
use Register::*;
impl Register {
    #[rustfmt::skip]
    fn emit_code(self, byte_size: u8) -> &'static str {
        assert!(byte_size == 1 || byte_size == 2 || byte_size == 4 || byte_size == 8);
        match (self, byte_size) {
            (AX, 1) => "%al",   (AX, 2) => "%ax",   (AX, 4) => "%eax",  (AX, 8) => "%rax",
            (DX, 1) => "%dl",   (DX, 2) => "%dx",   (DX, 4) => "%edx",  (DX, 8) => "%rdx",
            (DI, 1) => "%dil",  (DI, 2) => "%di",   (DI, 4) => "%edi",  (DI, 8) => "%rdi",
            (SI, 1) => "%sil",  (SI, 2) => "%si",   (SI, 4) => "%esi",  (SI, 8) => "%rsi",
            (R8, 1) => "%r8b",  (R8, 2) => "%r8w",  (R8, 4) => "%r8d",  (R8, 8) => "%r8",
            (R9, 1) => "%r9b",  (R9, 2) => "%r9w",  (R9, 4) => "%r9d",  (R9, 8) => "%r9",
            (R10,1) => "%r10b", (R10,2) => "%r10w", (R10,4) => "%r10d", (R10,8) => "%r10",
            (R11,1) => "%r11b", (R11,2) => "%r11w", (R11,4) => "%r11d", (R11,8) => "%r11",
            (CX, 1) => "%cl",   (CX, 2) => "%cx",   (CX, 4) => "%ecx",  (CX, 8) => "%rcx",
       /* stack pointer */      (SP, 2) => "%sp",   (SP, 4) => "%esp",  (SP, 8) => "%rsp",
       /* stack pointer */      (BP, 2) => "%bp",   (BP, 4) => "%ebp",  (BP, 8) => "%rbp",
            (XMM0, _) => "%xmm0",
            (XMM1, _) => "%xmm1",
            (XMM2, _) => "%xmm2",
            (XMM3, _) => "%xmm3",
            (XMM4, _) => "%xmm4",
            (XMM5, _) => "%xmm5",
            (XMM6, _) => "%xmm6",
            (XMM7, _) => "%xmm7",
            (XMM14, _) => "%xmm14",
            (XMM15, _) => "%xmm15",
            _ => unreachable!()
        }
    }

    fn is_xmm(self) -> bool {
        matches!(self, XMM0 | XMM1 | XMM2 | XMM3 | XMM4 | XMM5 | XMM6 | XMM7 | XMM14 | XMM15)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CondCode {
    E,
    NE,
    A,
    AE,
    B,
    BE,
    L,
    LE,
    G,
    GE,
    P,
}
use CondCode::*;
impl CondCode {
    fn emit_code(self) -> &'static str {
        match self {
            Self::E => "e",
            Self::NE => "ne",
            Self::A => "a",
            Self::AE => "ae",
            Self::B => "b",
            Self::BE => "be",
            Self::L => "l",
            Self::LE => "le",
            Self::G => "g",
            Self::GE => "ge",
            Self::P => "p",
        }
    }
}

// =========

const NEG_ZERO_NAME: &str = "L_neg0";
const UPPER_BOUND_NAME: &str = "L_upper_bound";
const UPPER_BOUND_F64: f64 = 9223372036854775808.0;
const UPPER_BOUND_INT: u64 = 9223372036854775808;

impl ir::Program {
    pub fn to_asm(&self) -> Program {
        // different order from the book
        let mut symbols = self.symbols.iter().map(|(k, v)| (k.clone(), v.clone().into())).collect();

        // just add it preemptively.
        let mut constants = vec![
            TopLevel::StaticConstant {
                name: NEG_ZERO_NAME.into(),
                init: StaticInit::Double(-0.0),
                alignment: 16,
            },
            TopLevel::StaticConstant {
                name: UPPER_BOUND_NAME.into(),
                init: StaticInit::Double(UPPER_BOUND_F64),
                alignment: 8,
            },
        ];

        let mut top_level: Vec<_> =
            self.top_level.iter().map(|tl| tl.to_asm(&symbols, &mut constants)).collect();

        // cudgle
        for cnst in &constants {
            let TopLevel::StaticConstant { name, init, .. } = cnst else {
                continue;
            };
            let type_ = match init {
                StaticInit::Int(_) | StaticInit::UInt(_) => Longword,
                StaticInit::Long(_) | StaticInit::ULong(_) => Quadword,
                StaticInit::Double(_) => Doubleword,
            };

            symbols.insert(
                name.clone(),
                BSymbol::Obj { type_, signed: false, is_static: true, constant: true },
            );
        }

        top_level.extend(constants);

        Program { top_level, symbols }
    }
}

impl ir::TopLevel {
    fn to_asm(&self, symbols: &Namespace<BSymbol>, consts: &mut Vec<TopLevel>) -> TopLevel {
        match self {
            Self::Function { name, global, params, body } => {
                let mut int_reg_args = Vec::new();
                let mut double_reg_args = Vec::new();
                let mut stack_args = Vec::new();

                for param in params {
                    let Some(BSymbol::Obj { type_: arg_ty, .. }) = symbols.get(param) else {
                        unreachable!()
                    };

                    match arg_ty {
                        Doubleword if double_reg_args.len() < 8 => double_reg_args.push(param),
                        Longword | Quadword if int_reg_args.len() < 6 => {
                            int_reg_args.push((param, *arg_ty));
                        }
                        Doubleword | Longword | Quadword => stack_args.push((param, *arg_ty)),
                    }
                }

                let instrs = int_reg_args
                    .into_iter()
                    .zip(ARG_REGISTERS)
                    .map(|(param, reg)| (param, Reg(reg, param.1.width())))
                    .chain(
                        double_reg_args
                            .into_iter()
                            .zip(DOUBLE_ARG_REGISTER)
                            .map(|(param, reg)| ((param, Doubleword), Reg(reg, 8))),
                    )
                    .chain(stack_args.into_iter().zip((16..).step_by(8).map(|d| Memory(BP, d))))
                    .map(|((param, arg_ty), reg)| {
                        Instr::Mov(arg_ty, reg.align_width(arg_ty), Pseudo(param.clone()))
                    })
                    .chain(body.iter().flat_map(|e| e.to_asm(symbols, consts)))
                    .collect();

                TopLevel::Function { name: name.clone(), instrs, stack_size: 0, global: *global }
            }
            Self::StaticVar { name, global, type_, init } => TopLevel::StaticVariable {
                name: name.clone(),
                global: *global,
                init: *init,
                alignment: match type_ {
                    Type::Int | Type::UInt => 4,
                    Type::Long | Type::ULong | Type::Pointer { .. } | Type::Double => 8,
                    Type::Func { .. } => unreachable!(),
                    Type::Array { .. } => todo!(),
                },
            },
        }
    }
}

impl ir::Instr {
    fn to_asm(&self, symbols: &Namespace<BSymbol>, consts: &mut Vec<TopLevel>) -> Vec<Instr> {
        match self {
            Self::Return(value) => {
                let (ty, _) = value.to_asm_type(symbols);
                vec![Instr::Mov(ty, value.to_asm(consts), ty.return_register()), Instr::Ret]
            }

            Self::SignExtend { src, dst } => vec![Instr::Movsx(src.to_asm(consts), dst.to_asm())],
            Self::Truncate { src, dst } => {
                vec![Instr::Mov(Longword, src.to_asm(consts), dst.to_asm())]
            }
            Self::ZeroExtend { src, dst } => {
                vec![Instr::MovZeroExtend(src.to_asm(consts), dst.to_asm())]
            }

            Self::IntToDouble { src, dst } => {
                vec![Instr::Cvtsi2sd(src.to_asm_type(symbols).0, src.to_asm(consts), dst.to_asm())]
            }
            Self::DoubleToInt { src, dst } => vec![Instr::Cvttsd2si(
                ir::Value::Var(dst.clone()).to_asm_type(symbols).0,
                src.to_asm(consts),
                dst.to_asm(),
            )],

            Self::UIntToDouble { src, dst } => {
                let (src_ty, _) = src.to_asm_type(symbols);
                match src_ty {
                    Doubleword => unreachable!(),
                    Longword => vec![
                        Instr::MovZeroExtend(src.to_asm(consts), Reg(R11, 8)),
                        Instr::Cvtsi2sd(Quadword, Reg(R11, 8), dst.to_asm()),
                    ],
                    Quadword => {
                        let label_1 = eco_format!("_cvt_{}", GEN.fetch_add(1, Relaxed));
                        let label_2 = eco_format!("_cvt_{}", GEN.fetch_add(1, Relaxed));

                        let src = src.to_asm(consts);
                        let dst = dst.to_asm();
                        vec![
                            Instr::Cmp(Quadword, Imm(0), src.clone()),
                            Instr::JmpCC(L, label_1.clone()),
                            Instr::Cvtsi2sd(Quadword, src.clone(), dst.clone()),
                            Instr::Jmp(label_2.clone()),
                            Instr::Label(label_1),
                            Instr::Mov(Quadword, src, Reg(R10, 8)),
                            Instr::Mov(Quadword, Reg(R10, 8), Reg(R11, 8)),
                            Instr::Unary(Operator::Shr, Quadword, Reg(R11, 8)),
                            Instr::Binary(Operator::And, Quadword, Imm(1), Reg(R10, 8)),
                            Instr::Binary(Operator::Or, Quadword, Reg(R10, 8), Reg(R11, 8)),
                            Instr::Cvtsi2sd(Quadword, Reg(R11, 8), dst.clone()),
                            Instr::Binary(Operator::Add, Doubleword, dst.clone(), dst),
                            Instr::Label(label_2),
                        ]
                    }
                }
            }
            Self::DoubleToUInt { src, dst } => {
                let (dst_ty, _) = ir::Value::Var(dst.clone()).to_asm_type(symbols);
                match dst_ty {
                    Doubleword => unreachable!(),
                    Longword => vec![
                        Instr::Cvttsd2si(Quadword, src.to_asm(consts), Reg(R10, 8)),
                        Instr::Mov(Longword, Reg(R10, 4), dst.to_asm()),
                    ],
                    Quadword => {
                        let label_1 = eco_format!("_cvt_{}", GEN.fetch_add(1, Relaxed));
                        let label_2 = eco_format!("_cvt_{}", GEN.fetch_add(1, Relaxed));

                        let src = src.to_asm(consts);
                        let dst = dst.to_asm();
                        vec![
                            Instr::Cmp(Doubleword, Data(UPPER_BOUND_NAME.into()), src.clone()),
                            Instr::JmpCC(AE, label_1.clone()),
                            Instr::Cvttsd2si(Quadword, src.clone(), dst.clone()),
                            Instr::Jmp(label_2.clone()),
                            Instr::Label(label_1),
                            Instr::Mov(Doubleword, src, Reg(XMM14, 8)),
                            Instr::Binary(
                                Operator::Sub,
                                Doubleword,
                                Data(UPPER_BOUND_NAME.into()),
                                Reg(XMM14, 8),
                            ),
                            Instr::Cvttsd2si(Quadword, Reg(XMM14, 8), dst.clone()),
                            Instr::Mov(Quadword, Imm(UPPER_BOUND_INT as i64), Reg(R10, 8)),
                            Instr::Binary(Operator::Add, Quadword, Reg(R10, 8), dst),
                            Instr::Label(label_2),
                        ]
                    }
                }
            }

            Self::Unary { op: ir::UnOp::Not, src, dst } => {
                let (src_ty, _) = src.to_asm_type(symbols);
                let Some(BSymbol::Obj { type_: dst_ty, .. }) = symbols.get(&dst.0) else {
                    unreachable!()
                };
                let nan = eco_format!("nan_{}", GEN.fetch_add(1, Relaxed));
                let dst = dst.to_asm();
                match src_ty {
                    Longword | Quadword => vec![
                        Instr::Cmp(src_ty, Imm(0), src.to_asm(consts)),
                        Instr::Mov(*dst_ty, Imm(0), dst.clone()),
                        Instr::SetCC(E, dst),
                    ],
                    Doubleword => vec![
                        Instr::Binary(Operator::Xor, Doubleword, Reg(XMM14, 8), Reg(XMM14, 8)),
                        Instr::Cmp(src_ty, src.to_asm(consts), Reg(XMM14, 8)),
                        Instr::Mov(*dst_ty, Imm(0), dst.clone()),
                        Instr::JmpCC(P, nan.clone()),
                        Instr::SetCC(E, dst),
                        Instr::Label(nan),
                    ],
                }
            }
            Self::Unary { op: ir::UnOp::Negate, src, dst }
                if src.to_asm_type(symbols).0 == Doubleword =>
            {
                let dst = dst.to_asm();
                vec![
                    Instr::Mov(Doubleword, src.to_asm(consts), dst.clone()),
                    Instr::Binary(Operator::Xor, Doubleword, Data(NEG_ZERO_NAME.into()), dst),
                ]
            }
            Self::Unary { op, src, dst } => {
                let (src_ty, _) = src.to_asm_type(symbols);
                let dst = dst.to_asm();
                vec![
                    Instr::Mov(src_ty, src.to_asm(consts), dst.clone()),
                    Instr::Unary(op.to_asm(), src_ty, dst),
                ]
            }
            Self::Binary { op, lhs, rhs, dst } => {
                let (src_ty, signed) = lhs.to_asm_type(symbols);
                match op {
                    ir::BinOp::Add
                    | ir::BinOp::Subtract
                    | ir::BinOp::Multiply
                    | ir::BinOp::BitAnd
                    | ir::BinOp::BitOr
                    | ir::BinOp::BitXor
                    | ir::BinOp::LeftShift
                    | ir::BinOp::RightShift => {
                        let dst = dst.to_asm();
                        vec![
                            Instr::Mov(src_ty, lhs.to_asm(consts), dst.clone()),
                            Instr::Binary(
                                op.to_asm(signed).unwrap_left(),
                                src_ty,
                                rhs.to_asm(consts),
                                dst,
                            ),
                        ]
                    }
                    ir::BinOp::Divide if matches!(src_ty, Doubleword) => {
                        // same code as previous section . not sure how to avoid duplication.
                        let dst = dst.to_asm();
                        vec![
                            Instr::Mov(src_ty, lhs.to_asm(consts), dst.clone()),
                            Instr::Binary(
                                op.to_asm(signed).unwrap_left(),
                                src_ty,
                                rhs.to_asm(consts),
                                dst,
                            ),
                        ]
                    }
                    ir::BinOp::Reminder if matches!(src_ty, Doubleword) => {
                        unreachable!("unable to modulo a double")
                    }

                    ir::BinOp::Divide | ir::BinOp::Reminder => {
                        let res = match op {
                            ir::BinOp::Divide => Register::AX,
                            ir::BinOp::Reminder => Register::DX,
                            _ => unreachable!(),
                        };
                        let mut instrs = vec![Instr::Mov(
                            src_ty,
                            lhs.to_asm(consts),
                            Reg(Register::AX, src_ty.width()),
                        )];
                        if signed {
                            instrs.extend([
                                Instr::Cdq(src_ty),
                                Instr::Idiv(src_ty, rhs.to_asm(consts)),
                            ]);
                        } else {
                            instrs.extend([
                                Instr::Mov(src_ty, Imm(0), Reg(DX, src_ty.width())),
                                Instr::Div(src_ty, rhs.to_asm(consts)),
                            ]);
                        }
                        instrs.push(Instr::Mov(src_ty, Reg(res, src_ty.width()), dst.to_asm()));

                        instrs
                    }

                    ir::BinOp::Equal
                    | ir::BinOp::NotEqual
                    | ir::BinOp::LessThan
                    | ir::BinOp::LessOrEqual
                    | ir::BinOp::GreaterThan
                    | ir::BinOp::GreaterOrEqual => {
                        let Some(BSymbol::Obj { type_: dst_ty, .. }) = symbols.get(&dst.0) else {
                            unreachable!()
                        };
                        let dst = dst.to_asm();
                        let counter = GEN.fetch_add(1, Relaxed);
                        let nan_label = eco_format!("nan_{}", counter);
                        let ian_label = eco_format!("ian_{}", counter);

                        match src_ty {
                            Doubleword if matches!(op, ir::BinOp::NotEqual) => vec![
                                Instr::Cmp(src_ty, rhs.to_asm(consts), lhs.to_asm(consts)),
                                Instr::Mov(*dst_ty, Imm(0), dst.clone()),
                                Instr::JmpCC(P, nan_label.clone()),
                                Instr::SetCC(NE, dst.clone()),
                                Instr::Jmp(ian_label.clone()),
                                Instr::Label(nan_label),
                                Instr::SetCC(E, dst),
                                Instr::Label(ian_label),
                            ],
                            Doubleword => vec![
                                Instr::Cmp(src_ty, rhs.to_asm(consts), lhs.to_asm(consts)),
                                Instr::Mov(*dst_ty, Imm(0), dst.clone()),
                                Instr::JmpCC(P, nan_label.clone()),
                                Instr::SetCC(op.to_asm(signed).unwrap_right(), dst),
                                Instr::Label(nan_label),
                            ],
                            _ => vec![
                                Instr::Cmp(src_ty, rhs.to_asm(consts), lhs.to_asm(consts)),
                                Instr::Mov(*dst_ty, Imm(0), dst.clone()),
                                Instr::SetCC(op.to_asm(signed).unwrap_right(), dst),
                            ],
                        }
                    }
                }
            }
            Self::Copy { src, dst } => {
                let (src_ty, _) = src.to_asm_type(symbols);
                vec![Instr::Mov(src_ty, src.to_asm(consts), dst.to_asm())]
            }
            Self::Jump { target } => vec![Instr::Jmp(target.clone())],
            Self::JumpIfZero { cond, target } => {
                let (cond_ty, _) = cond.to_asm_type(symbols);

                let counter = GEN.fetch_add(1, Relaxed);
                let nan_label = eco_format!("nan_{}", counter);

                match cond_ty {
                    Longword | Quadword => vec![
                        Instr::Cmp(cond_ty, Imm(0), cond.to_asm(consts)),
                        Instr::JmpCC(E, target.clone()),
                    ],
                    Doubleword => vec![
                        Instr::Binary(Operator::Xor, Doubleword, Reg(XMM15, 8), Reg(XMM15, 8)),
                        Instr::Cmp(Doubleword, cond.to_asm(consts), Reg(XMM15, 8)),
                        Instr::JmpCC(P, nan_label.clone()),
                        Instr::JmpCC(E, target.clone()),
                        Instr::Label(nan_label),
                    ],
                }
            }
            Self::JumpIfNotZero { cond, target } => {
                let (cond_ty, _) = cond.to_asm_type(symbols);

                let counter = GEN.fetch_add(1, Relaxed);
                let nan_label = eco_format!("nan_{}", counter);
                let ian_label = eco_format!("ian_{}", counter);

                match cond_ty {
                    Longword | Quadword => vec![
                        Instr::Cmp(cond_ty, Imm(0), cond.to_asm(consts)),
                        Instr::JmpCC(NE, target.clone()),
                    ],

                    Doubleword => vec![
                        Instr::Binary(Operator::Xor, Doubleword, Reg(XMM15, 8), Reg(XMM15, 8)),
                        Instr::Cmp(Doubleword, cond.to_asm(consts), Reg(XMM15, 8)),
                        Instr::JmpCC(P, nan_label.clone()),
                        Instr::JmpCC(NE, target.clone()),
                        Instr::Jmp(ian_label.clone()),
                        Instr::Label(nan_label),
                        Instr::JmpCC(E, target.clone()),
                        Instr::Label(ian_label),
                    ],
                }
            }
            Self::Label(name) => vec![Instr::Label(name.clone())],
            Self::FuncCall { name, args, dst } => {
                let Some(BSymbol::Obj { type_: dst_ty, .. }) = symbols.get(&dst.0) else {
                    unreachable!()
                };

                let mut int_reg_args = Vec::new();
                let mut double_reg_args = Vec::new();
                let mut stack_args = Vec::new();

                for value in args {
                    let (arg_ty, _) = value.to_asm_type(symbols);
                    let value = value.to_asm(consts);

                    match arg_ty {
                        Doubleword if double_reg_args.len() < 8 => double_reg_args.push(value),
                        Longword | Quadword if int_reg_args.len() < 6 => {
                            int_reg_args.push((value, arg_ty));
                        }
                        Doubleword | Longword | Quadword => stack_args.push((value, arg_ty)),
                    }
                }

                let stack_depth = stack_args.len();

                (stack_depth % 2 != 0)
                    .then_some(Instr::Binary(Operator::Sub, Quadword, Imm(8), Reg(SP, 8)))
                    .into_iter()
                    .chain(int_reg_args.into_iter().zip(ARG_REGISTERS).map(
                        |((op, arg_ty), reg)| Instr::Mov(arg_ty, op, Reg(reg, arg_ty.width())),
                    ))
                    .chain(
                        double_reg_args
                            .into_iter()
                            .zip(DOUBLE_ARG_REGISTER)
                            .map(|(op, reg)| Instr::Mov(Doubleword, op, Reg(reg, 8))),
                    )
                    .chain(stack_args.into_iter().rev().flat_map(|(op, arg_ty)| match op {
                        opp @ Imm(_) => vec![Instr::Push(opp)],
                        Reg(r, _) => vec![Instr::Push(Reg(r, 8))],
                        opp => {
                            let arg_ty = if arg_ty == Longword { Longword } else { Quadword };
                            vec![
                                Instr::Mov(arg_ty, opp, Reg(AX, arg_ty.width())),
                                Instr::Push(Reg(AX, 8)),
                            ]
                        }
                    }))
                    .chain([Instr::Call(name.clone())])
                    .chain((stack_depth > 0).then_some(Instr::Binary(
                        Operator::Add,
                        Quadword,
                        Imm(8 * stack_depth as i64
                            + (stack_depth % 2 != 0).then_some(8).unwrap_or_default()),
                        Reg(SP, 8),
                    )))
                    .chain([Instr::Mov(*dst_ty, dst_ty.return_register(), dst.to_asm())])
                    .collect()
            }
            Self::GetAddress { src, dst } => {
                let src = src.to_asm(consts);
                let dst = dst.to_asm();

                vec![Instr::Lea(src, dst)]
            }
            Self::Load { src_ptr, dst } => {
                let (dst_ty, _) = ir::Value::Var(dst.clone()).to_asm_type(symbols);
                let dst = dst.to_asm();

                vec![
                    Instr::Mov(Quadword, src_ptr.to_asm(consts), Reg(AX, 8)),
                    Instr::Mov(dst_ty, Memory(AX, 0), dst),
                ]
            }
            Self::Store { src, dst_ptr } => {
                let (src_ty, _) = src.to_asm_type(symbols);

                vec![
                    Instr::Mov(Quadword, dst_ptr.to_asm(), Reg(AX, 8)),
                    Instr::Mov(src_ty, src.to_asm(consts), Memory(AX, 0)),
                ]
            }
        }
    }
}

impl ir::Value {
    fn to_asm(&self, consts: &mut Vec<TopLevel>) -> Operand {
        match self {
            Self::Const(ast::Const::Int(i)) => Imm(i64::from(*i)),
            Self::Const(ast::Const::Long(i)) => Imm(*i),
            Self::Const(ast::Const::UInt(i)) => Imm(i64::from(*i)),
            Self::Const(ast::Const::ULong(i)) => Imm(*i as i64),
            // FLOAT CONSTANTS
            Self::Const(ast::Const::Double(i)) => {
                if let Some(TopLevel::StaticConstant { name, .. }) = consts.iter().find(|tl| {
                    matches!(tl, TopLevel::StaticConstant {
                            init: ast::StaticInit::Double(inner), ..
                        } if inner.to_bits() == i.to_bits())
                }) {
                    Data(name.clone())
                } else {
                    let name = eco_format!("L_c_{}", GEN.fetch_add(1, Relaxed));
                    let tl = TopLevel::StaticConstant {
                        name: name.clone(),
                        init: ast::StaticInit::Double(*i),
                        alignment: 8,
                    };
                    consts.push(tl);

                    Data(name)
                }
            }

            Self::Var(place) => place.to_asm(),
        }
    }
    fn to_asm_type(&self, symbols: &Namespace<BSymbol>) -> (AsmType, bool) {
        match self {
            Self::Const(c) => c.to_asm_type(),

            Self::Var(ir::Place(ident)) => match symbols.get(ident) {
                Some(BSymbol::Obj { type_, signed, .. }) => (*type_, *signed),
                e => unreachable!("{e:?}"),
            },
        }
    }
}
impl ast::Const {
    fn to_asm_type(self) -> (AsmType, bool) {
        match self {
            Self::Int(_) => (Longword, true),
            Self::UInt(_) => (Longword, false),
            Self::Long(_) => (Quadword, true),
            Self::ULong(_) => (Quadword, false),
            Self::Double(_) => (Doubleword, false),
        }
    }
}
impl ir::Place {
    fn to_asm(&self) -> Operand {
        Pseudo(self.0.clone())
    }
}
impl ir::UnOp {
    fn to_asm(self) -> Operator {
        match self {
            Self::Complement => Operator::Not,
            Self::Negate => Operator::Neg,
            Self::Not => unreachable!("Not is implemented otherwise"),
        }
    }
}
impl ir::BinOp {
    fn to_asm(self, signed: bool) -> Either<Operator, CondCode> {
        match self {
            Self::Add => Left(Operator::Add),
            Self::Subtract => Left(Operator::Sub),
            Self::Multiply => Left(Operator::Mul),
            Self::BitAnd => Left(Operator::And),
            Self::BitOr => Left(Operator::Or),
            Self::BitXor => Left(Operator::Xor),
            Self::LeftShift => Left(Operator::Shl),
            Self::RightShift if signed => Left(Operator::Sar),
            Self::RightShift => Left(Operator::Shr),
            Self::Divide => Left(Operator::DivDouble), // double division only

            Self::Reminder => {
                unreachable!("Divide and Reminder are implemented in other ways")
            }

            Self::Equal => Right(CondCode::E),
            Self::NotEqual => Right(CondCode::NE),
            Self::LessThan if signed => Right(CondCode::L),
            Self::LessThan => Right(CondCode::B),
            Self::LessOrEqual if signed => Right(CondCode::LE),
            Self::LessOrEqual => Right(CondCode::BE),
            Self::GreaterThan if signed => Right(CondCode::G),
            Self::GreaterThan => Right(CondCode::A),
            Self::GreaterOrEqual if signed => Right(CondCode::GE),
            Self::GreaterOrEqual => Right(CondCode::AE),
        }
    }
}

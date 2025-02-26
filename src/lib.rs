#![expect(unused)]

use std::{fs::read_to_string, io::Write, path::PathBuf};

mod asm;
mod ast;
mod lexer;
mod parser;
mod tac;
mod token;

use asm::Assembly;
use ast::Node;
use tac::Tac;

pub fn compile(input: &PathBuf, [lex, parse, tacky, codegen]: [bool; 4]) -> Result<PathBuf, u8> {
    let Ok(code) = read_to_string(input) else {
        eprintln!("Unable to read file.");
        return Err(2);
    };

    let Ok(tokens) = lexer::lex(&code) else {
        eprintln!("Failed to lex.");
        return Err(3);
    };

    if lex {
        eprintln!("{tokens:#?}");
        return Err(0);
    }

    let Ok(prgm) = parser::parse(&tokens) else {
        eprintln!("Failed to parse.");
        return Err(4);
    };

    if parse {
        eprintln!("{prgm:#?}");
        return Err(0);
    }

    let prgm = prgm.compile();

    if tacky {
        eprintln!("{prgm:#?}");
        return Err(0);
    }

    let mut prgm: asm::Program = prgm.to_asm();
    prgm.fixup_passes();

    if codegen {
        eprintln!("{prgm:#?}");

        // let mut buf = Vec::new();
        // prgm.emit_code(&mut buf);

        // let buf = String::from_utf8_lossy(&buf);
        // eprintln!("{buf}");

        return Err(0);
    }

    let output = input.with_extension("s");

    let Ok(mut output_file) = std::fs::File::create_new(&output) else {
        eprintln!("couldn't write to {}.", output.to_string_lossy());
        return Err(5);
    };

    prgm.emit_code(&mut output_file);
    _ = output_file.flush();

    Ok(output)
}

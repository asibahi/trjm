#![expect(unused)]

use std::{fs::read_to_string, io::Write, path::PathBuf};

mod asm;
mod ast;
mod lexer;
mod parser;
mod token;

use asm::Asm;
use ast::Node;

pub fn compile(input: &PathBuf, (lex, parse, codegen): (bool, bool, bool)) -> Result<PathBuf, u8> {
    let Ok(code) = read_to_string(input) else {
        eprintln!("Unable to read file.");
        return Err(2);
    };

    let Ok(tokens) = lexer::lex(&code) else {
        eprintln!("Failed to lex.");
        return Err(3);
    };

    if lex {
        eprintln!("{tokens:?}");
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

    let prgm: asm::Program = prgm.to_asm();

    if codegen {
        eprintln!("{prgm:#?}");

        let mut buf = Vec::new();
        prgm.emit_code(&mut buf);

        let buf = String::from_utf8_lossy(&buf);
        eprintln!("{buf}");

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

use std::{fs::read_to_string, io::Write, path::PathBuf};

mod assembly;
mod ast;
mod ir;
mod lexer;
mod parser;
mod token;

use assembly::{Assembly, ToAsm};

#[derive(Clone, Copy)]
pub enum Mode {
    Lex,
    Parse,
    Tac,
    Codegen,
    Validate,
    Assembly,
    ObjectFile,
    Executable,
}

pub fn compile(input: &PathBuf, mode: Mode) -> Result<PathBuf, u8> {
    let Ok(code) = read_to_string(input) else {
        eprintln!("Unable to read file.");
        return Err(2);
    };

    let Ok(tokens) = lexer::lex(&code) else {
        eprintln!("Failed to lex.");
        return Err(3);
    };

    if matches!(mode, Mode::Lex) {
        eprintln!("{tokens:#?}");
        return Err(0);
    }

    let prgm = match parser::parse(&tokens) {
        Ok(prgm) => prgm,
        Err(e) => {
            eprintln!("Failed to parse. {e:#?}");
            return Err(4);
        }
    };

    if matches!(mode, Mode::Parse) {
        eprintln!("{prgm:#?}");
        return Err(0);
    }

    let Some(prgm) = prgm.semantic_analysis() else {
        eprintln!("failed semantic analysis.");
        return Err(5);
    };

    if matches!(mode, Mode::Validate) {
        eprintln!("{prgm:#?}");
        return Err(0);
    }

    let prgm = prgm.compile();

    if matches!(mode, Mode::Tac) {
        eprintln!("{prgm:#?}");
        return Err(0);
    }

    let mut prgm: assembly::Program = prgm.to_asm();
    prgm.fixup_passes();

    if matches!(mode, Mode::Codegen) {
        eprintln!("{prgm:#?}");

        return Err(0);
    }

    if matches!(mode, Mode::Assembly) {
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

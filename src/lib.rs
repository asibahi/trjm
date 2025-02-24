#![expect(unused)]

use pico_args::Arguments;
use std::{
    fs::{read_to_string, remove_file},
    io::Write,
    path::PathBuf,
    process::{Command, exit},
};

mod asm;
mod ast;
mod lexer;
mod parser;
mod token;

use asm::Asm;
use ast::Node;

pub fn run() {
    let (mode, input) = parse();

    let preprocessed = preprocess(input);

    let compiled = compile(preprocessed, mode);

    if matches!(mode, (false, false, false)) {
        _ = assemble(compiled);
    }
}

type Mode = (bool, bool, bool);

fn parse() -> (Mode, PathBuf) {
    let mut args = Arguments::from_env();

    let mode = (
        args.contains("--lexer"),
        args.contains("--parse"),
        args.contains("--codegen"),
    );

    // let assembly_only = args.contains("-S");

    let Ok(input) = args.free_from_fn(validate_path) else {
        eprintln!("Input file required");
        exit(1);
    };
    (mode, input)
}

fn validate_path(s: &str) -> Result<PathBuf, &'static str> {
    if s.starts_with("-") {
        return Err("unknown argument.");
    }

    let path = PathBuf::from(s);
    if path.is_file() && path.exists() {
        Ok(path.to_path_buf())
    } else {
        Err("not a file.")
    }
}

fn preprocess(input: PathBuf) -> PathBuf {
    let output = input.with_extension("i");

    let status = Command::new("gcc")
        .arg("-E")
        .arg("-P")
        .arg(input)
        .arg("-o")
        .arg(&output)
        .status();

    if status.is_err() || status.is_ok_and(|s| !s.success()) {
        eprintln!("preprocessor failed");
        exit(1);
    }

    output
}

fn compile(input: PathBuf, (lex, parse, codegen): Mode) -> PathBuf {
    let Ok(code) = read_to_string(&input) else {
        eprintln!("Unable to read file.");
        exit(2);
    };

    let Ok(tokens) = lexer::lex(&code) else {
        eprintln!("Failed to lex.");
        exit(3);
    };

    if lex {
        eprintln!("{tokens:?}");
        exit(0)
    }

    let Ok(prgm) = parser::parse(tokens) else {
        eprintln!("Failed to parse.");
        exit(4);
    };

    if parse {
        eprintln!("{prgm:#?}");
        exit(0)
    }

    let prgm: asm::Program = prgm.to_asm();

    if codegen {
        eprintln!("{prgm:#?}");

        let mut buf = Vec::new();
        prgm.emit_code(&mut buf);

        let buf = String::from_utf8_lossy(&buf);
        eprintln!("{buf}");

        exit(0)
    }

    let output = input.with_extension("s");

    let Ok(mut output_file) = std::fs::File::create_new(&output) else {
        eprintln!("couldn't write to {}.", output.to_string_lossy());
        exit(5);
    };

    prgm.emit_code(&mut output_file);
    _ = output_file.flush();

    _ = remove_file(input);
    output
}

fn assemble(input: PathBuf) -> PathBuf {
    let output = input.with_extension("");

    let status = Command::new("gcc")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .status();

    _ = remove_file(input);

    if status.is_err() || status.is_ok_and(|s| !s.success()) {
        eprintln!("assembler failed");
        exit(1);
    }

    output
}

#![expect(unused)]

use std::{fs::read_to_string, io, path::PathBuf, process::exit};

mod ast;
mod lexer;
mod parser;
mod token;

pub fn run() {
    let mut args = pico_args::Arguments::from_env();

    let lex = args.contains("--lex");
    let prs = args.contains("--parse");

    let Ok(input) = args
        .free_from_fn(validate_path)
        .inspect_err(|e| eprintln!("Input file required. {e} found"))
    else {
        eprintln!("Unable to read file.");
        exit(2);
    };

    let Ok(tokens) = lexer::lex(&input) else {
        eprintln!("Failed to lex.");
        exit(3);
    };

    println!("{tokens:?}");

    if lex {
        exit(0)
    }

    let Ok(prgm) = parser::parse(tokens) else {
        eprintln!("Failed to parse.");
        exit(4);
    };

    println!("{prgm:#?}");

    if prs {
        exit(0)
    }
}

fn validate_path(s: &str) -> io::Result<String> {
    let path = PathBuf::from(s);
    read_to_string(path)
}

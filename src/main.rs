#![expect(unused)]

use std::{fs::read_to_string, path::PathBuf, process::exit};

mod lexer;
mod token;

fn main() {
    let mut args = pico_args::Arguments::from_env();

    let lex = args.contains("--lex");

    let input = match args.free_from_fn(validate_path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Input file required. {e} found");
            exit(1);
        }
    };

    let Ok(input) = read_to_string(input) else {
        eprintln!("Unable to read file.");
        exit(2);
    };

    let Ok(res) = lexer::lex(&input) else {
        eprintln!("Failed to lex.");
        exit(3);
    };

    if lex {
        exit(0)
    }
}

fn validate_path(s: &str) -> Result<PathBuf, &'static str> {
    let path = PathBuf::from(s);
    if path.is_file() {
        Ok(path.to_path_buf())
    } else {
        Err("not a file.")
    }
}

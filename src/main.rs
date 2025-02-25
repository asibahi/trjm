use pico_args::Arguments;
use std::{
    fs::remove_file,
    path::PathBuf,
    process::{Command, ExitCode},
};

fn main() -> ExitCode {
    let (mode, input) = match parse() {
        Ok(mi) => mi,
        Err(e) => return e,
    };

    let preprocessed = match preprocess(&input) {
        Ok(v) => v,
        Err(e) => return e,
    };

    let compiled = match trjm::compile(&preprocessed, mode) {
        Ok(c) => c,
        Err(e) => return ExitCode::from(e),
    };
    _ = remove_file(preprocessed);

    if matches!(mode, (false, false, false)) {
        let res = assemble(&compiled);
        _ = remove_file(&compiled);

        match res {
            Ok(_) => ExitCode::SUCCESS,
            Err(e) => e,
        }
    } else {
        ExitCode::SUCCESS
    }
}

type Mode = (bool, bool, bool);

fn parse() -> Result<(Mode, PathBuf), ExitCode> {
    let mut args = Arguments::from_env();

    let mode = (
        args.contains("--lex"),
        args.contains("--parse"),
        args.contains("--codegen"),
    );

    // let assembly_only = args.contains("-S");

    match args.free_from_fn(validate_path) {
        Ok(i) => Ok((mode, i)),
        Err(e) => {
            eprintln!("Input file required. {e}");
            Err(ExitCode::from(43))
        }
    }
}

fn validate_path(s: &str) -> Result<PathBuf, &'static str> {
    if s.starts_with('-') {
        return Err("unknown argument.");
    }

    let path = PathBuf::from(s);
    if path.is_file() && path.exists() {
        Ok(path.clone())
    } else {
        Err("not a file.")
    }
}

fn preprocess(input: &PathBuf) -> Result<PathBuf, ExitCode> {
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
        Err(ExitCode::from(42))
    } else {
        Ok(output)
    }
}

fn assemble(input: &PathBuf) -> Result<PathBuf, ExitCode> {
    let output = input.with_extension("");

    let status = Command::new("gcc")
        .arg(input)
        .arg("-o")
        .arg(&output)
        .status();

    if status.is_err() || status.is_ok_and(|s| !s.success()) {
        eprintln!("assembler failed");
        Err(ExitCode::from(65))
    } else {
        Ok(output)
    }
}

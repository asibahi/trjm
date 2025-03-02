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
        Err(e) => {
            _ = remove_file(preprocessed);
            return ExitCode::from(e);
        }
    };
    _ = remove_file(preprocessed);

    let res = match mode {
        trjm::Mode::ObjectFile => assemble(&compiled, true),
        trjm::Mode::Executable => assemble(&compiled, false),
        _ => return ExitCode::SUCCESS,
    };
    _ = remove_file(&compiled);

    match res {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => e,
    }
}

fn parse() -> Result<(trjm::Mode, PathBuf), ExitCode> {
    let mut args = Arguments::from_env();

    let mode = match () {
        // mutually exclusive
        _ if args.contains("--lex") => trjm::Mode::Lex,
        _ if args.contains("--parse") => trjm::Mode::Parse,
        _ if args.contains("--tacky") => trjm::Mode::Tac,
        _ if args.contains("--codegen") => trjm::Mode::Codegen,
        _ if args.contains("--validate") => trjm::Mode::Validate,
        _ if args.contains("--S") => trjm::Mode::Assembly,
        _ if args.contains("-c") => trjm::Mode::ObjectFile,
        _ => trjm::Mode::Executable,
    };

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
    if path.is_file() { Ok(path.clone()) } else { Err("not a file.") }
}

fn preprocess(input: &PathBuf) -> Result<PathBuf, ExitCode> {
    let output = input.with_extension("i");

    let status = Command::new("gcc").arg("-E").arg("-P").arg(input).arg("-o").arg(&output).status();

    if status.is_err() || status.is_ok_and(|s| !s.success()) {
        eprintln!("preprocessor failed");
        Err(ExitCode::from(42))
    } else {
        Ok(output)
    }
}

fn assemble(input: &PathBuf, object_file: bool) -> Result<PathBuf, ExitCode> {
    let output = if object_file { input.with_extension("o") } else { input.with_extension("") };

    let mut cmd = Command::new("gcc");
    if object_file {
        cmd.arg("-c");
    }
    let status = cmd.arg(input).arg("-o").arg(&output).status();

    if status.is_err() || status.is_ok_and(|s| !s.success()) {
        eprintln!("assembler failed");
        Err(ExitCode::from(65))
    } else {
        Ok(output)
    }
}

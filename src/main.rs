use pico_args::Arguments;
use std::{fs::remove_file, path::PathBuf, process::ExitCode};

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

    let res = match args.free_from_fn(validate_path) {
        Ok(i) => (mode, i),
        Err(e) => {
            eprintln!("Input file required. {e}");
            return Err(ExitCode::from(43));
        }
    };
    let rem = args.finish();
    if !rem.is_empty() {
        eprintln!("Unknwon additional arguments: {rem:?}");
        return Err(ExitCode::from(99));
    }

    Ok(res)
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

    let cmd =
        duct::cmd!("gcc", "-E", "-P", input, "-o", &output).unchecked().stderr_capture().run();
    match cmd {
        Ok(o) if !o.status.success() => {
            eprintln!("preprocessor failed:\n{}", String::from_utf8_lossy(&o.stdout));
            Err(ExitCode::from(41))
        }
        Err(e) => {
            eprintln!("preprocessor io error: {e}");
            Err(ExitCode::from(42))
        }
        Ok(_) => Ok(output),
    }
}

fn assemble(input: &PathBuf, object_file: bool) -> Result<PathBuf, ExitCode> {
    let (ext, obj) = if object_file { ("o", "-c") } else { ("", "") };

    let output = input.with_extension(ext);
    let cmd = duct::cmd!("gcc", obj, input, "-o", &output).unchecked().stderr_capture().run();

    match cmd {
        Ok(o) if !o.status.success() => {
            eprintln!("assembler failed:'n{}", String::from_utf8_lossy(&o.stdout));
            Err(ExitCode::from(66))
        }
        Err(e) => {
            eprintln!("assembler io error: {e}");
            Err(ExitCode::from(65))
        }
        Ok(_) => Ok(output),
    }
}

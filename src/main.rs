use pico_args::Arguments;
use std::{ffi::OsString, fs::remove_file, path::PathBuf, process::ExitCode};

fn main() -> ExitCode {
    let (mode, input, rem_args) = match parse() {
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
        trjm::Mode::ObjectFile => assemble(&compiled, true, rem_args),
        trjm::Mode::Executable => assemble(&compiled, false, rem_args),
        _ => return ExitCode::SUCCESS,
    };
    _ = remove_file(&compiled);

    match res {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => e,
    }
}

fn parse() -> Result<(trjm::Mode, PathBuf, Vec<OsString>), ExitCode> {
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

    let path = match args.free_from_fn(validate_path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Input file required. {e}");
            return Err(ExitCode::from(43));
        }
    };
    let rem = args.finish();

    Ok((mode, path, rem))
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
            eprintln!("preprocessor failed:\n{}", String::from_utf8_lossy(&o.stderr));
            Err(ExitCode::from(41))
        }
        Err(e) => {
            eprintln!("preprocessor io error: {e}");
            Err(ExitCode::from(42))
        }
        Ok(_) => Ok(output),
    }
}

fn assemble(
    input: &PathBuf,
    object_file: bool,
    extra_args: Vec<OsString>,
) -> Result<PathBuf, ExitCode> {
    let (ext, obj) = if object_file { ("o", "-c") } else { ("", "") };
    let extra_args = extra_args.join(std::ffi::OsStr::new(" "));

    let output = input.with_extension(ext);
    let cmd =
        duct::cmd!("gcc", obj, input, "-o", &output, extra_args).unchecked().stderr_capture().run();

    match cmd {
        Ok(o) if !o.status.success() => {
            eprintln!("assembler failed:\n{}", String::from_utf8_lossy(&o.stderr));
            Err(ExitCode::from(66))
        }
        Err(e) => {
            eprintln!("assembler io error: {e}");
            Err(ExitCode::from(65))
        }
        Ok(_) => Ok(output),
    }
}

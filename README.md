# `trjm`, a C17 subset compiler

This compiler is basically me following, chapter by chapter, [Writing a C Compiler](https://nostarch.com/writing-c-compiler) by Nora Sandler.

As of this writing, it implements everything up to chapter 13 (doubles), including extra credit (and also binary, octal, and hex number literals!) It passes every test up to that point in [the official test suite](https://github.com/nlsandler/writing-a-c-compiler-tests).

The lexer and parser are implemented using `nom` and `nom-language`. But the rest of hte code base follows the book closely.

The compiler is self contained. All you need to do to install it and run is to usual Cargo tooling. Depends on a C compiler being used with the `gcc` command.

It supports the following flags:

- `--parse` to get the parsed output of the AST.
- `--tacky` to get the IR.
- `--S` to get the assembly being generated.

Output is to `stdout` or it saves a file. 

Works on x86 emulation in MacOS.

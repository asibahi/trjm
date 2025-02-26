use crate::token::Token;
use nom::{
    AsChar, Finish, IResult, Parser,
    branch::alt,
    bytes::{tag, take_while},
    character::{complete::i32, multispace0, satisfy},
    combinator::{all_consuming, peek, recognize},
    multi::many,
    sequence::{preceded, terminated},
};

type LexError<'s> = ();
// type LexError<'s> = (&'s str, nom::ErrorKind);
type Emit = nom::OutputM<nom::Emit, nom::Emit, nom::Complete>;
type Check = nom::OutputM<nom::Check, nom::Emit, nom::Complete>;

pub fn lex(i: &str) -> Result<Vec<Token>, LexError<'_>> {
    let tokens = alt((
        decrement,
        hyphen,
        tilde,
        semicolon,
        paren_open,
        paren_close,
        brace_open,
        brace_close,
        identifier,
        number,
    ));

    let (i, _) = multispace0().process::<Check>(i).finish()?;

    all_consuming(many(1.., tokens))
        .process::<Emit>(i)
        .finish()
        .map(|t| t.1)
}

macro_rules! token {
    ($func:ident, $body:expr) => {
        fn $func(i: &str) -> IResult<&str, Token, LexError<'_>> {
            let (i, token) = $body.process::<Emit>(i)?;
            let (i, _) = multispace0().process::<Check>(i)?;

            Ok((i, token))
        }
    };
    ($func:ident, $token:ident, $tag:literal) => {
        token!($func, tag($tag).map(|_| Token::$token));
    };
}

// order of these is important in the main function.
token!(decrement, Decrement, "--");
token!(hyphen, Hyphen, "-");

token!(tilde, Tilde, "~");
token!(semicolon, Semicolon, ";");
token!(paren_open, ParenOpen, "(");
token!(paren_close, ParenClose, ")");
token!(brace_open, BraceOpen, "{");
token!(brace_close, BraceClose, "}");
token!(
    identifier,
    recognize(preceded(
        satisfy(|c| c == '_' || c.is_alpha()),
        take_while(|c: char| c == '_' || c.is_alphanum()),
    ))
    .map(|s: &str| match s {
        "int" => Token::Int,
        "void" => Token::Void,
        "return" => Token::Return,
        s => Token::Ident(s.into()),
    })
);
token!(
    number,
    terminated(i32, peek(satisfy(|c| c != '_' && !c.is_alpha()))).map(Token::NumberLiteral)
);

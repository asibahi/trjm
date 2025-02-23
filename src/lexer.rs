use ecow::EcoString;
use nom::{
    AsChar, Finish, IResult, Input, Parser,
    branch::alt,
    bytes::{complete::take_until, tag, take_while, take_while1},
    character::{complete::usize, multispace0, satisfy},
    combinator::{all_consuming, peek, recognize},
    multi::many,
    sequence::{delimited, preceded, terminated},
};

use crate::token::*;

pub(crate) fn lex(i: &str) -> Result<Vec<Token>, ()> {
    let tokens = alt((
        semicolon,
        paren_open,
        paren_close,
        brace_open,
        brace_close,
        identifier,
        number,
        comments,
    ));

    let (i, _) = multispace0().parse_complete(i).finish()?;

    let (_, ret) = all_consuming(many(1.., tokens))
        .parse_complete(i)
        .finish()?;

    Ok(ret)
}

macro_rules! token {
    ($func:ident, $body:expr) => {
        fn $func(i: &str) -> IResult<&str, Token, ()> {
            let (i, token) = $body.parse_complete(i)?;
            let (i, _) = multispace0().parse_complete(i)?;

            Ok((i, token))
        }
    };
    ($func:ident, $token:ident, $tag:literal) => {
        token!($func, tag($tag).map(|_| Token::$token));
    };
}

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
        s => Token::Identifier(s.into()),
    })
);
token!(
    number,
    terminated(usize, peek(satisfy(|c| c != '_' && !c.is_alpha()))).map(Token::NumberLiteral)
);
token!(
    comments,
    alt((
        preceded(tag("//"), take_until("\n")),
        delimited(tag("/*"), take_until("*/"), tag("*/"))
    ))
    .map(|_| Token::Comments)
);

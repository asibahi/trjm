#![warn(dead_code)]

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
        alt((decrement, hyphen)),
        alt((incrmeent, plus)),
        //
        alt((asterisk, forward_slash, percent, tilde)),
        //
        alt((dbl_equal, equal)),
        alt((bang_equal, bang)),
        //
        alt((dbl_ambersand, dbl_pipe, ambersand, pipe, caret)),
        //
        alt((left_shift, less_equal, less_than)),
        alt((right_shift, greater_equal, greater_than)),
        //
        question_mark,
        colon,
        //
        alt((semicolon, paren_open, paren_close, brace_open, brace_close)),
        //
        identifier,
        number,
    ));

    all_consuming(many(1.., tokens))
        .process::<Emit>(i.trim())
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

// order of these is important
token!(decrement, Decrement, "--");
token!(hyphen, Hyphen, "-");

token!(incrmeent, Increment, "++");
token!(plus, Plus, "+");

token!(asterisk, Astrisk, "*");
token!(forward_slash, ForwardSlash, "/");
token!(percent, Percent, "%");
token!(tilde, Tilde, "~");

token!(dbl_equal, DblEqual, "==");
token!(equal, Equal, "=");

token!(bang_equal, BangEqual, "!=");
token!(bang, Bang, "!");

token!(dbl_ambersand, DblAmbersand, "&&");
token!(dbl_pipe, DblPipe, "||");
token!(ambersand, Ambersand, "&");
token!(pipe, Pipe, "|");
token!(caret, Caret, "^");

token!(question_mark, QuestionMark, "?");
token!(colon, Colon, ":");

// should later be into a famly
token!(left_shift, LeftShift, "<<");
token!(less_equal, LessEqual, "<=");
token!(less_than, LessThan, "<");

token!(right_shift, RightShift, ">>");
token!(greater_equal, GreaterEqual, ">=");
token!(greater_than, GreaterThan, ">");

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

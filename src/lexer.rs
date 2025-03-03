#![warn(dead_code)]

use crate::token::Token;
use nom::{
    AsChar, Finish, IResult, Parser,
    branch::alt,
    bytes::{tag, take_while},
    character::{
        complete::{bin_digit1, hex_digit1, i32, oct_digit0},
        multispace0, satisfy,
    },
    combinator::{all_consuming, cut, not, recognize},
    multi::many,
    sequence::{preceded, terminated},
};

type LexError<'s> = ();
// type LexError<'s> = (&'s str, nom::ErrorKind);

pub fn lex(i: &str) -> Result<Vec<Token>, LexError<'_>> {
    let tokens = alt((
        alt((decrement, hyphen_equal, hyphen)),
        alt((incrmeent, plus_equal, plus)),
        //
        alt((
            asterisk_equal,
            asterisk,
            forward_slash_equal,
            forward_slash,
            percent_equal,
            percent,
            tilde,
        )),
        //
        alt((dbl_equal, equal)),
        alt((bang_equal, bang)),
        //
        alt((
            dbl_ambersand,
            ambersand_equal,
            dbl_pipe,
            pipe_equal,
            ambersand,
            pipe,
            caret_equal,
            caret,
        )),
        //
        alt((
            left_shift_equal,
            right_shift_equal,
            //
            left_shift,
            right_shift,
            //
            less_equal,
            greater_equal,
            //
            less_than,
            greater_than,
        )),
        //
        question_mark,
        colon,
        //
        alt((semicolon, comma, paren_open, paren_close, brace_open, brace_close)),
        //
        keyword_or_identifier,
        number,
    ));

    all_consuming(many(1.., tokens)).parse_complete(i.trim()).finish().map(|t| t.1)
}

macro_rules! token {
    ($func:ident, $body:expr) => {
        fn $func(i: &str) -> IResult<&str, Token, LexError<'_>> {
            terminated($body, multispace0()).parse_complete(i)
        }
    };
    ($func:ident, $token:ident, $tag:literal) => {
        token!($func, tag($tag).map(|_| Token::$token));
    };
}

// order of these is important
token!(decrement, DblHyphen, "--");
token!(hyphen_equal, HyphenEqual, "-=");
token!(hyphen, Hyphen, "-");

token!(incrmeent, DblPlus, "++");
token!(plus_equal, PlusEqual, "+=");
token!(plus, Plus, "+");

token!(asterisk, Astrisk, "*");
token!(asterisk_equal, AstriskEqual, "*=");

token!(forward_slash, ForeSlash, "/");
token!(forward_slash_equal, ForeSlashEqual, "/=");

token!(percent, Percent, "%");
token!(percent_equal, PercentEqual, "%=");

token!(tilde, Tilde, "~");

token!(dbl_equal, DblEqual, "==");
token!(equal, Equal, "=");

token!(bang_equal, BangEqual, "!=");
token!(bang, Bang, "!");

token!(ambersand, Ambersand, "&");
token!(dbl_ambersand, DblAmbersand, "&&");
token!(ambersand_equal, AmbersandEqual, "&=");

token!(pipe, Pipe, "|");
token!(dbl_pipe, DblPipe, "||");
token!(pipe_equal, PipeEqual, "|=");

token!(caret, Caret, "^");
token!(caret_equal, CaretEqual, "^=");

token!(question_mark, QMark, "?");
token!(colon, Colon, ":");

// should later be into a famly
token!(left_shift_equal, LeftShiftEqual, "<<=");
token!(left_shift, LeftShift, "<<");
token!(less_equal, LessEqual, "<=");
token!(less_than, LessThan, "<");

token!(right_shift_equal, RightShiftEqual, ">>=");
token!(right_shift, RightShift, ">>");
token!(greater_equal, GreaterEqual, ">=");
token!(greater_than, GreaterThan, ">");

token!(semicolon, Semicolon, ";");
token!(comma, Comma, ",");
token!(paren_open, ParenOpen, "(");
token!(paren_close, ParenClose, ")");
token!(brace_open, BraceOpen, "{");
token!(brace_close, BraceClose, "}");

token!(
    keyword_or_identifier,
    recognize(preceded(
        satisfy(|c| c == '_' || c.is_alpha()),
        take_while(|c: char| c == '_' || c.is_alphanum()),
    ))
    .map(|s: &str| match s {
        "int" => Token::Int,
        "void" => Token::Void,
        "return" => Token::Return,
        "if" => Token::If,
        "else" => Token::Else,
        "goto" => Token::GoTo,
        "do" => Token::Do,
        "while" => Token::While,
        "for" => Token::For,
        "break" => Token::Break,
        "continue" => Token::Continue,
        "switch" => Token::Switch,
        "case" => Token::Case,
        "default" => Token::Default,
        s => Token::Ident(s.into()),
    })
);

macro_rules! radix (
    ($tag: expr, $parser:ident, $radix:literal) =>{
        preceded($tag, cut($parser))
        .map(|s: &str| i32::from_str_radix(s, $radix).unwrap_or_default())
    }
);

token!(
    number,
    terminated(
        alt((
            radix!(tag("0x").or(tag("0X")), hex_digit1, 16),
            radix!(tag("0b").or(tag("0B")), bin_digit1, 2),
            radix!(tag("0"), oct_digit0, 8),
            i32,
        )),
        not(satisfy(|c| c == '_' || c.is_alphanum())),
    )
    .map(Token::NumberLiteral)
);

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    #[test_case("0x0" => 0)]
    #[test_case("0x10" => 16)]
    #[test_case("0" => 0)]
    #[test_case("010" => 8)]
    #[test_case("10" => 10)]
    #[test_case("0b0" => 0)]
    #[test_case("0b10" => 2)]
    #[test_case("0b10-0xFFF" => 2)]
    #[test_case("0b10;int" => 2)]
    #[test_case("0x0g" => panics "illegal literal")]
    #[test_case("_0x0" => panics "illegal literal")]
    #[test_case("pubg" => panics "illegal literal")]
    #[test_case("0x10g" => panics "illegal literal")]
    #[test_case("0b" => panics "illegal literal")]
    #[test_case("0g" => panics "illegal literal")]
    #[test_case("0_" => panics "illegal literal")]
    #[test_case("010b" => panics "illegal literal")]
    #[test_case("0109" => panics "illegal literal")]
    #[test_case("10b" => panics "illegal literal")]
    #[test_case("10_" => panics "illegal literal")]
    #[test_case("0b0b" => panics "illegal literal")]
    #[test_case("0b10b" => panics "illegal literal")]
    #[test_case("0b10_" => panics "illegal literal")]
    fn any(i: &str) -> i32 {
        number(i).expect("illegal literal").1.unwrap_number().unwrap()
    }
}

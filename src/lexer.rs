#![warn(dead_code)]

use ecow::EcoString as Ecow;
use nom::{
    AsChar, Finish, IResult, Input, Parser,
    branch::alt,
    bytes::{tag, tag_no_case, take_while},
    character::{
        complete::{bin_digit1, hex_digit1, oct_digit0, u64},
        multispace0, satisfy,
    },
    combinator::{all_consuming, cut, not, recognize, success},
    multi::many,
    sequence::{preceded, terminated},
};
use std::iter::{Cloned, Enumerate};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Token {
    // Stuff
    Ident(Ecow),
    IntLit(i32),
    LongLit(i64),
    UIntLit(u32),
    ULongLit(u64),

    // types
    Int,
    Long,
    Signed,
    Unsigned,

    // Keywords
    Void, // also doubtful
    Return,
    If,
    Else,
    GoTo,
    Do,
    While,
    For,
    Break,
    Continue,
    Switch,
    Case,
    Default,
    Static,
    Extern,

    // Operators
    Plus,      // +
    PlusEqual, // +=
    DblPlus,   // ++

    Astrisk,
    AstriskEqual,

    ForeSlash,
    ForeSlashEqual,

    Percent,
    PercentEqual,

    Tilde,

    Hyphen,      // -
    HyphenEqual, // -=
    DblHyphen,   // --

    DblEqual,
    Equal,
    Bang,
    BangEqual,

    Ambersand,
    DblAmbersand,
    AmbersandEqual,

    Pipe,
    DblPipe,
    PipeEqual,

    Caret,
    CaretEqual,

    LessThan,     // <
    GreaterThan,  // >
    LessEqual,    // <=
    GreaterEqual, // >=

    LeftShift,       // <<
    LeftShiftEqual,  // <<=
    RightShift,      // >>
    RightShiftEqual, // >>=

    QMark,
    Colon,

    // Punctuation
    ParenOpen,
    ParenClose,
    BraceOpen,
    BraceClose,
    Semicolon,
    Comma,
}

impl Token {
    pub fn unwrap_ident(&self) -> Option<Ecow> {
        if let Token::Ident(s) = self { Some(s.clone()) } else { None }
    }
}

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
        "long" => Token::Long,
        "signed" => Token::Signed,
        "unsigned" => Token::Unsigned,
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
        "static" => Token::Static,
        "extern" => Token::Extern,
        s => Token::Ident(s.into()),
    })
);

macro_rules! radix (
    ($tag: expr, $parser:ident, $radix:literal) =>{
        preceded($tag, cut($parser))
        .map(|s: &str| u64::from_str_radix(s, $radix).unwrap_or_default())
    }
);

#[derive(Debug, Clone, Copy)]
enum IntType {
    Unsigned,
    UnsignedLong,
    Long,
    Unknown,
}

token!(
    number,
    alt((
        radix!(tag("0x").or(tag("0X")), hex_digit1, 16),
        radix!(tag("0b").or(tag("0B")), bin_digit1, 2),
        radix!(tag("0"), oct_digit0, 8),
        u64,
    ))
    .and(terminated(
        alt((
            tag_no_case("ul").or(tag_no_case("lu")).map(|_| IntType::UnsignedLong),
            tag_no_case("l").map(|_| IntType::Long),
            tag_no_case("u").map(|_| IntType::Unsigned),
            success(IntType::Unknown)
        )),
        not(satisfy(|c| c == '_' || c.is_alphanum())),
    ))
    .map(|(lit, ty)| match ty {
        // maybe this should be divorced from lexing into parsing.
        IntType::Unsigned => match u32::try_from(lit) {
            Ok(i) => Token::UIntLit(i),
            Err(_) => Token::ULongLit(lit),
        },
        IntType::UnsignedLong => Token::ULongLit(lit),
        IntType::Long => Token::LongLit(lit as i64),
        IntType::Unknown => match i32::try_from(lit) {
            Ok(i) => Token::IntLit(i),
            Err(_) => Token::LongLit(lit as i64),
        },
    })
);

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    // longs
    #[test_case("0x0L" => 0)]
    #[test_case("0x10L" => 16)]
    #[test_case("0L" => 0)]
    #[test_case("010L" => 8)]
    #[test_case("10l" => 10)]
    #[test_case("8L" => 8)]
    #[test_case("0b0L" => 0)]
    #[test_case("0b10L" => 2)]
    #[test_case("0b10L-0xFFF" => 2)]
    #[test_case("0b10L;int" => 2)]
    // ints
    #[test_case("0x0" => 0)]
    #[test_case("0x10" => 16)]
    #[test_case("0" => 0)]
    #[test_case("010" => 8)]
    #[test_case("10" => 10)]
    #[test_case("0b0" => 0)]
    #[test_case("0b10" => 2)]
    #[test_case("0b10-0xFFF" => 2)]
    #[test_case("0b10;int" => 2)]
    // invalid
    #[test_case("8Ll" => panics "illegal literal")]
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
    fn number_test(i: &str) -> i64 {
        match number(i).expect("illegal literal").1 {
            Token::IntLit(i) => i64::from(i),
            Token::LongLit(i) => i,
            _ => unreachable!(),
        }
    }
}

// ======

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Tokens<'s>(pub &'s [Token]);

impl<'s> From<&'s [Token]> for Tokens<'s> {
    fn from(value: &'s [Token]) -> Self {
        Tokens(value)
    }
}

// stolen wholesale from nom's impl of &[u8]
impl<'s> Input for Tokens<'s> {
    type Item = Token;
    type Iter = Cloned<std::slice::Iter<'s, Token>>;
    type IterIndices = Enumerate<Self::Iter>;

    fn input_len(&self) -> usize {
        self.0.len()
    }

    fn take(&self, index: usize) -> Self {
        Self(&self.0[0..index])
    }

    fn take_from(&self, index: usize) -> Self {
        Self(&self.0[index..])
    }

    fn take_split(&self, index: usize) -> (Self, Self) {
        let (prefix, suffix) = self.0.split_at(index);
        (Self(suffix), Self(prefix))
    }

    fn position<P>(&self, predicate: P) -> Option<usize>
    where
        P: Fn(Self::Item) -> bool,
    {
        self.0.iter().position(|t| predicate(t.clone()))
    }

    fn iter_elements(&self) -> Self::Iter {
        self.0.iter().cloned()
    }

    fn iter_indices(&self) -> Self::IterIndices {
        self.iter_elements().enumerate()
    }

    fn slice_index(&self, count: usize) -> Result<usize, nom::Needed> {
        if self.0.len() >= count { Ok(count) } else { Err(nom::Needed::new(count - self.0.len())) }
    }
}

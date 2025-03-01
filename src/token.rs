use std::iter::{Cloned, Enumerate};

use ecow::EcoString;
use nom::Input;

#[warn(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Token {
    // Stuff
    Ident(EcoString),
    NumberLiteral(i32),

    // Keywords
    Int,  // doubtful
    Void, // also doubtful
    Return,
    If,
    Else,
    GoTo,

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
}

impl Token {
    pub fn unwrap_ident(&self) -> Option<EcoString> {
        if let Token::Ident(s) = self { Some(s.clone()) } else { None }
    }

    pub fn unwrap_number(&self) -> Option<i32> {
        if let Token::NumberLiteral(v) = self { Some(*v) } else { None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Tokens<'s>(pub &'s [Token]);

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

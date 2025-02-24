use crate::{
    ast::{Expr, FuncDef, Ident, Program, Stmt},
    token::{Token, Tokens},
};
use nom::{
    Finish, IResult, Parser,
    bytes::take,
    combinator::{all_consuming, verify},
    sequence::delimited,
};

pub fn parse(tokens: &[Token]) -> Result<Program, ()> {
    all_consuming(parse_program)
        .parse_complete(Tokens(tokens))
        .finish()
        .map(|t| t.1)
}

fn parse_program(i: Tokens<'_>) -> IResult<Tokens<'_>, Program, ()> {
    let (i, func_def) = parse_func(i)?;

    Ok((i, Program(func_def)))
}

macro_rules! tag_token {
    ($token:pat) => {
        verify(take(1u8), |t: &Tokens| matches!(t.0[0], $token))
    };
}

fn parse_func(i: Tokens<'_>) -> IResult<Tokens<'_>, FuncDef, ()> {
    let (i, _) = tag_token!(Token::Int).parse_complete(i)?;
    let (i, name) = tag_token!(Token::Ident(_))
        .map_opt(|t: Tokens<'_>| t.0[0].unwrap_ident().map(Ident))
        .parse_complete(i)?;

    let (i, _) = tag_token!(Token::ParenOpen).parse_complete(i)?;
    let (i, _) = tag_token!(Token::Void).parse_complete(i)?;
    let (i, _) = tag_token!(Token::ParenClose).parse_complete(i)?;

    let (i, _) = tag_token!(Token::BraceOpen).parse_complete(i)?;

    let (i, body) = parse_stmt(i)?;

    let (i, _) = tag_token!(Token::BraceClose).parse_complete(i)?;

    let func_def = FuncDef { name, body };

    Ok((i, func_def))
}

fn parse_stmt(i: Tokens<'_>) -> IResult<Tokens<'_>, Stmt, ()> {
    // parses return statement
    delimited(
        tag_token!(Token::Return),
        parse_expr,
        tag_token!(Token::Semicolon),
    )
    .map(|e| Stmt::Return(Box::new(e)))
    .parse_complete(i)
}

fn parse_expr(i: Tokens<'_>) -> IResult<Tokens<'_>, Expr, ()> {
    // parse constant expression
    tag_token!(Token::NumberLiteral(_))
        .map_opt(|t: Tokens<'_>| t.0[0].unwrap_number().map(Expr::ConstInt))
        .parse_complete(i)
}

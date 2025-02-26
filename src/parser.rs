use crate::{
    ast::*,
    token::{Token, Tokens},
};
use ecow::EcoString;
use nom::{
    Finish, IResult, Parser,
    branch::alt,
    bytes::take,
    combinator::{all_consuming, verify},
    sequence::delimited,
};

type ParseError<'s> = ();
// type ParseError<'s> = (Tokens<'s>, nom::ErrorKind);
type Emit = nom::OutputM<nom::Emit, nom::Emit, nom::Complete>;
type Check = nom::OutputM<nom::Check, nom::Emit, nom::Complete>;

pub fn parse(tokens: &[Token]) -> Result<Program, ParseError<'_>> {
    all_consuming(parse_program)
        .parse_complete(Tokens(tokens))
        .finish()
        .map(|t| t.1)
}

fn parse_program(i: Tokens<'_>) -> IResult<Tokens<'_>, Program, ParseError<'_>> {
    let (i, func_def) = parse_func(i)?;

    Ok((i, Program(func_def)))
}

macro_rules! tag_token {
    ($token:pat) => {
        verify(take(1u8), |t: &Tokens| matches!(t.0[0], $token))
    };
}

fn parse_func(i: Tokens<'_>) -> IResult<Tokens<'_>, FuncDef, ParseError<'_>> {
    let (i, _) = tag_token!(Token::Int).process::<Check>(i)?;
    let (i, name) = tag_token!(Token::Ident(_))
        .map_opt(|t: Tokens<'_>| t.0[0].unwrap_ident())
        .process::<Emit>(i)?;

    let (i, _) = tag_token!(Token::ParenOpen).process::<Check>(i)?;
    let (i, _) = tag_token!(Token::Void).process::<Check>(i)?;
    let (i, _) = tag_token!(Token::ParenClose).process::<Check>(i)?;

    let (i, _) = tag_token!(Token::BraceOpen).process::<Check>(i)?;

    let (i, body) = parse_stmt(i)?;

    let (i, _) = tag_token!(Token::BraceClose).process::<Check>(i)?;

    let func_def = FuncDef { name, body };

    Ok((i, func_def))
}

fn parse_stmt(i: Tokens<'_>) -> IResult<Tokens<'_>, Stmt, ParseError<'_>> {
    // parses return statement
    delimited(
        tag_token!(Token::Return),
        parse_expr,
        tag_token!(Token::Semicolon),
    )
    .map(|e| Stmt::Return(Box::new(e)))
    .process::<Emit>(i)
}

fn parse_expr(i: Tokens<'_>) -> IResult<Tokens<'_>, Expr, ParseError<'_>> {
    let const_expr = tag_token!(Token::NumberLiteral(_))
        .map_opt(|t: Tokens<'_>| t.0[0].unwrap_number().map(Expr::ConstInt));

    let unop = (parse_unop, parse_expr).map(|(u, e)| Expr::Unary(u, Box::new(e)));

    let grp_expr = delimited(
        tag_token!(Token::ParenOpen),
        parse_expr,
        tag_token!(Token::ParenClose),
    );

    alt((const_expr, unop, grp_expr)).process::<Emit>(i)
}

fn parse_unop(i: Tokens<'_>) -> IResult<Tokens<'_>, UnaryOp, ParseError<'_>> {
    // parse constant expression
    let neg = tag_token!(Token::Hyphen).map(|_| UnaryOp::Negate);
    let compl = tag_token!(Token::Tilde).map(|_| UnaryOp::Complement);

    alt((neg, compl)).process::<Emit>(i)
}

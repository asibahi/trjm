use crate::{
    ast::*,
    token::{Token, Tokens},
};
use ecow::EcoString;
use either::Either::{Left, Right};
use nom::{
    Finish, IResult, Parser,
    branch::alt,
    bytes::take,
    combinator::{all_consuming, opt, verify},
    multi::many,
    sequence::{delimited, preceded, separated_pair, terminated},
};
use nom_language::precedence::{Assoc, Operation, binary_op, precedence, unary_op};

// type ParseError<'s> = ();
type ParseError<'s> = (Tokens<'s>, nom::error::ErrorKind);
type Emit = nom::OutputM<nom::Emit, nom::Emit, nom::Complete>;
type Check = nom::OutputM<nom::Check, nom::Emit, nom::Complete>;

pub fn parse(tokens: &[Token]) -> Result<Program, ParseError<'_>> {
    all_consuming(parse_program).parse_complete(Tokens(tokens)).finish().map(|t| t.1)
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

    let (i, body) = parse_block.process::<Emit>(i)?;

    let func_def = FuncDef { name, body };

    Ok((i, func_def))
}

fn parse_block(i: Tokens<'_>) -> IResult<Tokens<'_>, Block, ParseError<'_>> {
    delimited(
        tag_token!(Token::BraceOpen),
        many(.., alt((parse_decl.map(BlockItem::D), parse_stmt.map(BlockItem::S)))),
        tag_token!(Token::BraceClose),
    )
    .map(Block)
    .process::<Emit>(i)
}

fn parse_ident(i: Tokens<'_>) -> IResult<Tokens<'_>, EcoString, ParseError<'_>> {
    tag_token!(Token::Ident(_)).map_opt(|t: Tokens<'_>| t.0[0].unwrap_ident()).process::<Emit>(i)
}

fn parse_decl(i: Tokens<'_>) -> IResult<Tokens<'_>, Decl, ParseError<'_>> {
    let (i, _) = tag_token!(Token::Int).process::<Check>(i)?;
    let (i, name) = parse_ident(i)?;
    let (i, init) = opt(preceded(tag_token!(Token::Equal), parse_expr)).process::<Emit>(i)?;
    let (i, _) = tag_token!(Token::Semicolon).process::<Check>(i)?;
    let ret = Decl { name, init };

    Ok((i, ret))
}

fn parse_stmt(i: Tokens<'_>) -> IResult<Tokens<'_>, Stmt, ParseError<'_>> {
    let ret = delimited(tag_token!(Token::Return), parse_expr, tag_token!(Token::Semicolon))
        .map(Stmt::Return);
    let expr = terminated(parse_expr, tag_token!(Token::Semicolon)).map(Stmt::Expression);

    let if_else = preceded(
        tag_token!(Token::If),
        (
            delimited(tag_token!(Token::ParenOpen), parse_expr, tag_token!(Token::ParenClose)),
            parse_stmt.map(Box::new),
            opt(preceded(tag_token!(Token::Else), parse_stmt).map(Box::new)),
        )
            .map(|(cond, then, else_)| Stmt::If { cond, then, else_ }),
    );

    let compound = parse_block.map(Stmt::Compound);

    let goto = delimited(tag_token!(Token::GoTo), parse_ident, tag_token!(Token::Semicolon))
        .map(Stmt::GoTo);
    let label = (terminated(parse_ident, tag_token!(Token::Colon)), parse_stmt)
        .map(|(n, s)| Stmt::Label(n, Box::new(s)));

    let break_ = terminated(tag_token!(Token::Break), tag_token!(Token::Semicolon))
        .map(|_| Stmt::Break(None));
    let continue_ = terminated(tag_token!(Token::Continue), tag_token!(Token::Semicolon))
        .map(|_| Stmt::Continue(None));

    let while_ = preceded(
        (tag_token!(Token::While), tag_token!(Token::ParenOpen)),
        separated_pair(parse_expr, tag_token!(Token::ParenClose), parse_stmt.map(Box::new)),
    )
    .map(|(cond, body)| Stmt::While { cond, body, label: None });

    let do_while_ = delimited(
        tag_token!(Token::Do),
        separated_pair(
            parse_stmt.map(Box::new),
            (tag_token!(Token::While), tag_token!(Token::ParenOpen)),
            parse_expr,
        ),
        (tag_token!(Token::ParenClose), tag_token!(Token::Semicolon)),
    )
    .map(|(body, cond)| Stmt::DoWhile { body, cond, label: None });

    let for_ = preceded(
        (tag_token!(Token::For), tag_token!(Token::ParenOpen)),
        (
            alt((
                parse_decl.map(Left), // already includes semicolon
                terminated(opt(parse_expr).map(Right), tag_token!(Token::Semicolon)),
            )),
            terminated(opt(parse_expr), tag_token!(Token::Semicolon)),
            terminated(opt(parse_expr), tag_token!(Token::ParenClose)),
            parse_stmt.map(Box::new),
        ),
    )
    .map(|(init, cond, post, body)| Stmt::For { init, cond, post, body, label: None });

    let null = tag_token!(Token::Semicolon).map(|_| Stmt::Null);

    alt((
        ret, expr, if_else, compound, goto, label, break_, continue_, while_, do_while_, for_, null,
    ))
    .process::<Emit>(i)
}

enum BinKind {
    Typical(BinaryOp),
    Ternary(Expr),
    Assignment,
    CompoundAssignment(BinaryOp),
}
use BinKind::*;

macro_rules! binop {
    ($prec:literal, $token:ident, $binop:ident) => {
        binary_op(
            $prec,
            Assoc::Left,
            tag_token!(Token::$token).map(|_| BinaryOp::$binop).map(Typical),
        )
    };
    (= $token:ident, $binop:ident) => {
        binary_op(
            14,
            Assoc::Right,
            tag_token!(Token::$token).map(|_| BinaryOp::$binop).map(CompoundAssignment),
        )
    };
}

fn parse_expr(i: Tokens<'_>) -> IResult<Tokens<'_>, Expr, ParseError<'_>> {
    // precedence reference:
    // https://en.cppreference.com/w/c/language/operator_precedence

    precedence(
        // prefix
        alt((
            unary_op(2, tag_token!(Token::Hyphen).map(|_| UnaryOp::Negate)),
            unary_op(2, tag_token!(Token::Tilde).map(|_| UnaryOp::Complement)),
            unary_op(2, tag_token!(Token::Bang).map(|_| UnaryOp::Not)),
            unary_op(2, tag_token!(Token::Plus).map(|_| UnaryOp::Plus)),
            // chapter 5 extra credit
            unary_op(2, tag_token!(Token::DblPlus).map(|_| UnaryOp::IncPre)),
            unary_op(2, tag_token!(Token::DblHyphen).map(|_| UnaryOp::DecPre)),
        )),
        // postfix // chapter 5 extra credit
        alt((
            unary_op(1, tag_token!(Token::DblPlus).map(|_| UnaryOp::IncPost)),
            unary_op(1, tag_token!(Token::DblHyphen).map(|_| UnaryOp::DecPost)),
        )),
        // binary
        alt((
            alt((
                binop!(4, Plus, Add),
                binop!(4, Hyphen, Subtract),
                binop!(3, Astrisk, Multiply),
                binop!(3, ForeSlash, Divide),
                binop!(3, Percent, Reminder),
            )),
            // chapter 3 extra credit
            alt((
                binop!(5, LeftShift, LeftShift),
                binop!(5, RightShift, RightShift),
                binop!(8, Ambersand, BitAnd),
                binop!(9, Caret, BitXor),
                binop!(10, Pipe, BitOr),
                // chapter 6
                binary_op(
                    13,
                    Assoc::Right,
                    delimited(tag_token!(Token::QMark), parse_expr, tag_token!(Token::Colon))
                        .map(Ternary),
                ),
            )),
            // chapter 4
            alt((
                binop!(7, DblEqual, Equal),
                binop!(7, BangEqual, NotEqual),
                binop!(11, DblAmbersand, And),
                binop!(12, DblPipe, Or),
                binop!(6, LessThan, LessThan),
                binop!(6, LessEqual, LessOrEqual),
                binop!(6, GreaterThan, GreaterThan),
                binop!(6, GreaterEqual, GreaterOrEqual),
            )),
            // chapter 5
            alt((
                binary_op(14, Assoc::Right, tag_token!(Token::Equal).map(|_| Assignment)),
                // chapter 5 extra credit
                binop!(= PlusEqual, Add),
                binop!(= HyphenEqual, Subtract),
                binop!(= AstriskEqual, Multiply),
                binop!(= ForeSlashEqual, Divide),
                binop!(= PercentEqual, Reminder),
                binop!(= AmbersandEqual, BitAnd),
                binop!(= PipeEqual, BitOr),
                binop!(= CaretEqual, BitXor),
                binop!(= LeftShiftEqual, LeftShift),
                binop!(= RightShiftEqual, RightShift),
            )),
        )),
        // operands or factors
        alt((
            // const
            tag_token!(Token::NumberLiteral(_))
                .map_opt(|t: Tokens<'_>| t.0[0].unwrap_number().map(Expr::ConstInt)),
            // group
            delimited(tag_token!(Token::ParenOpen), parse_expr, tag_token!(Token::ParenClose)),
            //variable
            tag_token!(Token::Ident(_))
                .map_opt(|t: Tokens<'_>| t.0[0].unwrap_ident())
                .map(Expr::Var),
        )),
        // fold
        |op| -> Result<Expr, ()> {
            Ok(match op {
                Operation::Prefix(op, exp) | Operation::Postfix(exp, op) => {
                    Expr::Unary(op, Box::new(exp))
                }
                Operation::Binary(lhs, Typical(op), rhs) => {
                    Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
                }
                Operation::Binary(lhs, CompoundAssignment(op), rhs) => {
                    Expr::CompoundAssignment { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
                }
                Operation::Binary(lhs, Assignment, rhs) => {
                    Expr::Assignemnt(Box::new(lhs), Box::new(rhs))
                }
                Operation::Binary(lhs, Ternary(op), rhs) => Expr::Conditional {
                    cond: Box::new(lhs),
                    then: Box::new(op),
                    else_: Box::new(rhs),
                },
            })
        },
    )(i)
}

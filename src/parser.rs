use crate::{
    ast::*,
    token::{Token, Tokens},
};
use nom::{
    Finish, IResult, Parser,
    branch::alt,
    bytes::take,
    combinator::{all_consuming, opt, verify},
    error::context,
    multi::many,
    sequence::{delimited, preceded, terminated},
};
use nom_language::precedence::{Assoc, Operation, binary_op, precedence, unary_op};

// type ParseError<'s> = ();
type ParseError<'s> = (Tokens<'s>, nom::error::ErrorKind);
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
        context(
            stringify!($token),
            verify(take(1u8), |t: &Tokens| matches!(t.0[0], $token)),
        )
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

    let (i, body) = many(.., parse_block_item).process::<Emit>(i)?;

    let (i, _) = tag_token!(Token::BraceClose).process::<Check>(i)?;

    let func_def = FuncDef { name, body };

    Ok((i, func_def))
}

fn parse_block_item(i: Tokens<'_>) -> IResult<Tokens<'_>, BlockItem, ParseError<'_>> {
    alt((
        context("decl", parse_decl.map(BlockItem::D)),
        context("stmt", parse_stmt.map(BlockItem::S)),
    ))
    .process::<Emit>(i)
}

fn parse_decl(i: Tokens<'_>) -> IResult<Tokens<'_>, Decl, ParseError<'_>> {
    let (i, _) = tag_token!(Token::Int).process::<Check>(i)?;
    let (i, name) = tag_token!(Token::Ident(_))
        .map_opt(|t: Tokens<'_>| t.0[0].unwrap_ident())
        .process::<Emit>(i)?;
    let (i, init) = opt(preceded(tag_token!(Token::Equal), parse_expr)).process::<Emit>(i)?;
    let (i, _) = tag_token!(Token::Semicolon).process::<Check>(i)?;
    let ret = Decl { name, init };

    Ok((i, ret))
}

fn parse_stmt(i: Tokens<'_>) -> IResult<Tokens<'_>, Stmt, ParseError<'_>> {
    let ret = delimited(
        tag_token!(Token::Return),
        parse_expr,
        tag_token!(Token::Semicolon),
    )
    .map(Stmt::Return);
    let expr = terminated(parse_expr, tag_token!(Token::Semicolon)).map(Stmt::Expression);
    let null = tag_token!(Token::Semicolon).map(|_| Stmt::Null);

    alt((ret, expr, null)).process::<Emit>(i)
}

enum BinKind {
    Typical(BinaryOp),
    Ternary(Expr),
    Assignment,
    CompoundAssignment(BinaryOp),
}
use BinKind::*;

macro_rules! binop {
    ($token:pat, $binop:expr) => {
        tag_token!($token).map(|_| $binop).map(Typical)
    };
    (= $token:pat, $binop:expr) => {
        tag_token!($token).map(|_| $binop).map(CompoundAssignment)
    };
}

#[allow(clippy::too_many_lines)]
fn parse_expr(i: Tokens<'_>) -> IResult<Tokens<'_>, Expr, ParseError<'_>> {
    // precedence reference:
    // https://en.cppreference.com/w/c/language/operator_precedence

    // Prefix expressions
    let neg = tag_token!(Token::Hyphen).map(|_| UnaryOp::Negate);
    let compl = tag_token!(Token::Tilde).map(|_| UnaryOp::Complement);
    let not = tag_token!(Token::Bang).map(|_| UnaryOp::Not);
    let plus = tag_token!(Token::Plus).map(|_| UnaryOp::Plus);

    // chapter 5 extra credit
    let inc_pre = tag_token!(Token::DblPlus).map(|_| UnaryOp::IncPre);
    let dec_pre = tag_token!(Token::DblHyphen).map(|_| UnaryOp::DecPre);

    // Postfix chapter 5 extra credit
    let inc_post = tag_token!(Token::DblPlus).map(|_| UnaryOp::IncPost);
    let dec_post = tag_token!(Token::DblHyphen).map(|_| UnaryOp::DecPost);

    // Infix expressions
    let add = binop!(Token::Plus, BinaryOp::Add);
    let sub = binop!(Token::Hyphen, BinaryOp::Subtract);
    let mul = binop!(Token::Astrisk, BinaryOp::Multiply);
    let div = binop!(Token::ForwardSlash, BinaryOp::Divide);
    let rem = binop!(Token::Percent, BinaryOp::Reminder);

    // chapter 3 extra credit
    let bit_and = binop!(Token::Ambersand, BinaryOp::BitAnd);
    let bit_or = binop!(Token::Pipe, BinaryOp::BitOr);
    let bit_xor = binop!(Token::Caret, BinaryOp::BitXor);

    let shl = binop!(Token::LeftShift, BinaryOp::LeftShift);
    let shr = binop!(Token::RightShift, BinaryOp::RightShift);

    // chapter 4
    let eq = binop!(Token::DblEqual, BinaryOp::Equal);
    let ne = binop!(Token::BangEqual, BinaryOp::NotEqual);
    let and = binop!(Token::DblAmbersand, BinaryOp::And);
    let or = binop!(Token::DblPipe, BinaryOp::Or);

    let lt = binop!(Token::LessThan, BinaryOp::LessThan);
    let le = binop!(Token::LessEqual, BinaryOp::LessOrEqual);
    let gt = binop!(Token::GreaterThan, BinaryOp::GreaterThan);
    let ge = binop!(Token::GreaterEqual, BinaryOp::GreaterOrEqual);

    //chapter 5
    let assign = tag_token!(Token::Equal).map(|_| Assignment);

    // chapter 5 extra credit
    let add_assign = binop!(= Token::PlusEqual, BinaryOp::Add);
    let sub_assign = binop!(= Token::HyphenEqual, BinaryOp::Subtract);
    let mul_assign = binop!(= Token::AstriskEqual, BinaryOp::Multiply);
    let div_assign = binop!(= Token::ForwardSlashEqual, BinaryOp::Divide);
    let rem_assign = binop!(= Token::PercentEqual, BinaryOp::Reminder);

    let and_assign = binop!(= Token::AmbersandEqual, BinaryOp::BitAnd);
    let or_assign = binop!(= Token::PipeEqual, BinaryOp::BitOr);
    let xor_assign = binop!(= Token::CaretEqual, BinaryOp::BitXor);
    let shl_assign = binop!(= Token::LeftShiftEqual, BinaryOp::LeftShift);
    let shr_assign = binop!(= Token::RightShiftEqual, BinaryOp::RightShift);

    // chapter 6
    let ternary = delimited(
        tag_token!(Token::QuestionMark),
        parse_expr,
        tag_token!(Token::Colon),
    )
    .map(Ternary);

    // Operand expressions _ or factors
    let const_expr = tag_token!(Token::NumberLiteral(_))
        .map_opt(|t: Tokens<'_>| t.0[0].unwrap_number().map(Expr::ConstInt));
    let grp_expr = delimited(
        tag_token!(Token::ParenOpen),
        parse_expr,
        tag_token!(Token::ParenClose),
    );
    let ident = tag_token!(Token::Ident(_))
        .map_opt(|t: Tokens<'_>| t.0[0].unwrap_ident())
        .map(Expr::Var);

    // where does the ternary fit in?

    precedence(
        // prefix
        alt((
            unary_op(2, neg),
            unary_op(2, compl),
            unary_op(2, not),
            unary_op(2, plus),
            unary_op(2, inc_pre),
            unary_op(2, dec_pre),
        )),
        // postfix
        alt((unary_op(1, inc_post), unary_op(1, dec_post))),
        // binary
        alt((
            alt((
                binary_op(3, Assoc::Left, mul),
                binary_op(3, Assoc::Left, div),
                binary_op(3, Assoc::Left, rem),
                binary_op(4, Assoc::Left, add),
                binary_op(4, Assoc::Left, sub),
            )),
            // chapter 3 extra credit
            alt((
                binary_op(5, Assoc::Left, shl),
                binary_op(5, Assoc::Left, shr),
                binary_op(8, Assoc::Left, bit_and),
                binary_op(9, Assoc::Left, bit_xor),
                binary_op(10, Assoc::Left, bit_or),
                binary_op(13, Assoc::Right, ternary),
            )),
            // chapter 4
            alt((
                binary_op(7, Assoc::Left, eq),
                binary_op(7, Assoc::Left, ne),
                binary_op(11, Assoc::Left, and),
                binary_op(12, Assoc::Left, or),
                binary_op(6, Assoc::Left, le),
                binary_op(6, Assoc::Left, lt),
                binary_op(6, Assoc::Left, ge),
                binary_op(6, Assoc::Left, gt),
            )),
            // chapter 5
            binary_op(14, Assoc::Right, assign),
            // chapter 5 extra credit
            alt((
                binary_op(14, Assoc::Right, add_assign),
                binary_op(14, Assoc::Right, sub_assign),
                binary_op(14, Assoc::Right, mul_assign),
                binary_op(14, Assoc::Right, div_assign),
                binary_op(14, Assoc::Right, rem_assign),
                binary_op(14, Assoc::Right, shl_assign),
                binary_op(14, Assoc::Right, shr_assign),
                binary_op(14, Assoc::Right, and_assign),
                binary_op(14, Assoc::Right, or_assign),
                binary_op(14, Assoc::Right, xor_assign),
            )),
        )),
        // operand
        alt((
            const_expr, // const
            grp_expr,   // grp
            ident,      // variable
        )),
        // fold
        |op| match op {
            Operation::Prefix(op, exp) | Operation::Postfix(exp, op) => {
                Ok::<_, ()>(Expr::Unary(op, Box::new(exp)))
            }
            Operation::Binary(lhs, Typical(op), rhs) => Ok(Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }),
            Operation::Binary(lhs, CompoundAssignment(op), rhs) => Ok(Expr::CompoundAssignment {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }),

            Operation::Binary(lhs, Assignment, rhs) => {
                Ok(Expr::Assignemnt(Box::new(lhs), Box::new(rhs)))
            }
            Operation::Binary(lhs, Ternary(op), rhs) => Ok(Expr::Conditional {
                cond: Box::new(lhs),
                then: Box::new(op),
                else_: Box::new(rhs),
            }),
        },
    )(i)
}

use crate::{
    ast::*,
    lexer::{Token, Tokens},
};
use ecow::EcoString as Ecow;
use either::Either::{self, Left, Right};
use nom::{
    Finish, IResult, Parser,
    branch::alt,
    bytes::take,
    combinator::{all_consuming, complete, iterator, not, opt, verify},
    error,
    multi::{many, separated_list0, separated_list1},
    sequence::{delimited, preceded, separated_pair, terminated},
};
use nom_language::precedence::{Assoc, Binary, Operation, Unary, binary_op, precedence, unary_op};

type ParseError<'s> = ();
// type ParseError<'s> = (Tokens<'s>, nom::error::ErrorKind);

pub fn parse(tokens: &[Token]) -> Result<Program, ParseError<'_>> {
    all_consuming(parse_program).parse_complete(Tokens(tokens)).finish().map(|t| t.1)
}

fn parse_program(i: Tokens<'_>) -> IResult<Tokens<'_>, Program, ParseError<'_>> {
    many(1.., parse_decl).map(Program::new).parse_complete(i)
}

fn parse_decl(i: Tokens<'_>) -> IResult<Tokens<'_>, Decl, ParseError<'_>> {
    parse_var_decl.map(Decl::Var).or(parse_func_decl.map(Decl::Func)).parse_complete(i)
}

macro_rules! tag_token {
    // weird counting stuff
    (@$t:pat) => { 1 };
    (@c $($token:pat),+) => { (0usize $(+ tag_token!(@$token))+) };

    ($($token:pat),+) => {
        verify(
            take((tag_token!(@c $($token),+)   )),
            |t: &Tokens| matches!(t.0[0..tag_token!(@c $($token),+)],[$($token),+])
        )
    };
    ($($token:pat),+ => $map:expr) => {
        tag_token!($($token),+).map(|_| $map)
    };
}

fn parse_type(i: Tokens<'_>) -> IResult<Tokens<'_>, Type, ParseError<'_>> {
    // this is clearly wrong for more types.
    terminated(
        alt((
            tag_token!(Token::Int, Token::Long => Type::Long),
            tag_token!(Token::Long, Token::Int => Type::Long),
            tag_token!(Token::Long => Type::Long),
            tag_token!(Token::Int => Type::Int),
        )),
        not(tag_token!(Token::Int)),
    )
    .parse_complete(i)
}

fn parse_specifiers(i: Tokens<'_>) -> IResult<Tokens<'_>, (Type, StorageClass), ParseError<'_>> {
    let mut iter = iterator(
        i,
        // nom issue #1835
        complete(alt((
            tag_token!(Token::Static => StorageClass::Static).map(Left),
            tag_token!(Token::Extern => StorageClass::Extern).map(Left),
            tag_token!(Token::Long => Token::Long).map(Right),
            tag_token!(Token::Int => Token::Int).map(Right),
        ))),
    );

    let (sc, ty) = iter.by_ref().partition::<Vec<_>, _>(Either::is_left);

    let (i, ()) = iter.finish()?;

    if sc.len() > 1 {
        #[allow(clippy::unit_arg)]
        return Err(nom::Err::Error(error::make_error(i, error::ErrorKind::TooLarge)));
    }

    let sc = sc.first().map_or(StorageClass::None, |e| e.clone().unwrap_left());

    let ty = ty.iter().map(|e| e.clone().unwrap_right()).collect::<Vec<_>>();
    let (_, ty) = parse_type(Tokens::from(&ty[..]))?;

    Ok((i, (ty, sc)))
}

fn parse_var_decl(i: Tokens<'_>) -> IResult<Tokens<'_>, VarDecl, ParseError<'_>> {
    let (i, (ty, sc)) = parse_specifiers.parse_complete(i)?;

    terminated(
        (parse_ident, opt(preceded(tag_token!(Token::Equal), parse_expr))),
        tag_token!(Token::Semicolon),
    )
    .map(|(name, init)| VarDecl { name, init, sc, var_type: ty.clone() })
    .parse_complete(i)
}

fn parse_func_decl(i: Tokens<'_>) -> IResult<Tokens<'_>, FuncDecl, ParseError<'_>> {
    let (i, (ret, sc)) = parse_specifiers.parse_complete(i)?;

    let (i, name) = parse_ident.parse_complete(i)?;

    let (i, params) = delimited(
        tag_token!(Token::ParenOpen),
        alt((
            tag_token!(Token::Void => Vec::new()),
            separated_list1(tag_token!(Token::Comma), parse_type.and(parse_ident)),
        )),
        tag_token!(Token::ParenClose),
    )
    .parse_complete(i)?;

    let (tps, params) = params.into_iter().unzip();

    let fun_type = Type::Func { params: tps, ret: Box::new(ret) };

    let (i, body) =
        parse_block.map(Some).or(tag_token!(Token::Semicolon => None)).parse_complete(i)?;

    let func_def = FuncDecl { name, params, body, sc, fun_type };

    Ok((i, func_def))
}

fn parse_block(i: Tokens<'_>) -> IResult<Tokens<'_>, Block, ParseError<'_>> {
    delimited(
        tag_token!(Token::BraceOpen),
        many(.., parse_decl.map(BlockItem::D).or(parse_stmt.map(BlockItem::S))),
        tag_token!(Token::BraceClose),
    )
    .map(Block)
    .parse_complete(i)
}

fn parse_ident(i: Tokens<'_>) -> IResult<Tokens<'_>, Ecow, ParseError<'_>> {
    tag_token!(Token::Ident(_)).map_opt(|t: Tokens<'_>| t.0[0].unwrap_ident()).parse_complete(i)
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
    let label = terminated(parse_ident, tag_token!(Token::Colon))
        .and(parse_stmt)
        .map(|(n, s)| Stmt::Label(n, Box::new(s)));

    let break_ = tag_token!(Token::Break, Token::Semicolon => Stmt::Break(None));
    let continue_ = tag_token!(Token::Continue, Token::Semicolon => Stmt::Continue(None));

    let while_ = preceded(
        tag_token!(Token::While, Token::ParenOpen),
        separated_pair(parse_expr, tag_token!(Token::ParenClose), parse_stmt.map(Box::new)),
    )
    .map(|(cond, body)| Stmt::While { cond, body, label: None });

    let do_while_ = delimited(
        tag_token!(Token::Do),
        separated_pair(
            parse_stmt.map(Box::new),
            tag_token!(Token::While, Token::ParenOpen),
            parse_expr,
        ),
        tag_token!(Token::ParenClose, Token::Semicolon),
    )
    .map(|(body, cond)| Stmt::DoWhile { body, cond, label: None });

    let for_ = preceded(
        tag_token!(Token::For, Token::ParenOpen),
        (
            alt((
                parse_var_decl.map(Left), // already includes semicolon
                terminated(opt(parse_expr).map(Right), tag_token!(Token::Semicolon)),
            )),
            terminated(opt(parse_expr), tag_token!(Token::Semicolon)),
            terminated(opt(parse_expr), tag_token!(Token::ParenClose)),
            parse_stmt.map(Box::new),
        ),
    )
    .map(|(init, cond, post, body)| Stmt::For { init, cond, post, body, label: None });

    let switch = preceded(
        tag_token!(Token::Switch, Token::ParenOpen),
        separated_pair(parse_expr, tag_token!(Token::ParenClose), parse_stmt),
    )
    .map(|(ctrl, body)| Stmt::Switch {
        ctrl,
        body: Box::new(body),
        label: None,
        cases: Vec::new(),
    });
    let case = preceded(
        tag_token!(Token::Case),
        separated_pair(parse_expr, tag_token!(Token::Colon), parse_stmt),
    )
    .map(|(cnst, body)| Stmt::Case { cnst, body: Box::new(body), label: None });

    let dflt = preceded(tag_token!(Token::Default, Token::Colon), parse_stmt)
        .map(|body| Stmt::Default { body: Box::new(body), label: None });

    let null = tag_token!(Token::Semicolon => Stmt::Null);

    alt((
        ret, expr, if_else, compound, goto, label, break_, continue_, while_, do_while_, for_,
        switch, case, dflt, null,
    ))
    .parse_complete(i)
}

enum BinKind {
    Typical(BinaryOp),
    Ternary(TypedExpr),
    Assignment,
    CompoundAssignment(BinaryOp),
}
use BinKind::*;

macro_rules! binop {
    ($prec:literal, $token:ident, $binop:ident) => {
        binary_op(
            $prec,
            Assoc::Left,
            tag_token!(Token::$token => BinaryOp::$binop).map(Typical),
        )
    };
    (= $token:ident, $binop:ident) => {
        binary_op(
            14,
            Assoc::Right,
            tag_token!(Token::$token => BinaryOp::$binop).map(CompoundAssignment),
        )
    };
}

fn parse_prefixes(
    i: Tokens<'_>,
) -> IResult<Tokens<'_>, Unary<Either<UnaryOp, Type>, i32>, ParseError<'_>> {
    alt((
        unary_op(2, tag_token!(Token::Hyphen => Left(UnaryOp::Negate))),
        unary_op(2, tag_token!(Token::Tilde => Left(UnaryOp::Complement))),
        unary_op(2, tag_token!(Token::Bang => Left(UnaryOp::Not))),
        unary_op(2, tag_token!(Token::Plus => Left(UnaryOp::Plus))),
        // chapter 5 extra credit
        unary_op(2, tag_token!(Token::DblPlus => Left(UnaryOp::IncPre))),
        unary_op(2, tag_token!(Token::DblHyphen => Left(UnaryOp::DecPre))),
        // chapter 11 cast operation
        unary_op(
            2,
            delimited(tag_token!(Token::ParenOpen), parse_type, tag_token!(Token::ParenClose))
                .map(Right),
        ),
    ))
    .parse_complete(i)
}

fn parse_postfixes(i: Tokens<'_>) -> IResult<Tokens<'_>, Unary<UnaryOp, i32>, ParseError<'_>> {
    // chapter 5 extra credit

    alt((
        unary_op(1, tag_token!(Token::DblPlus => UnaryOp::IncPost)),
        unary_op(1, tag_token!(Token::DblHyphen => UnaryOp::DecPost)),
    ))
    .parse_complete(i)
}

fn parse_infixes(i: Tokens<'_>) -> IResult<Tokens<'_>, Binary<BinKind, i32>, ParseError<'_>> {
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
            binary_op(14, Assoc::Right, tag_token!(Token::Equal => Assignment)),
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
    ))
    .parse_complete(i)
}

fn parse_operands(i: Tokens<'_>) -> IResult<Tokens<'_>, TypedExpr, ParseError<'_>> {
    alt((
        // function call
        parse_ident
            .and(delimited(
                tag_token!(Token::ParenOpen),
                separated_list0(tag_token!(Token::Comma), parse_expr),
                tag_token!(Token::ParenClose),
            ))
            .map(|(name, args)| Expr::FuncCall { name, args }.dummy_typed()),
        // const
        tag_token!(Token::IntLiteral(_) | Token::LongLiteral(_)).map_opt(|t: Tokens<'_>| {
            match t.0[0] {
                Token::IntLiteral(i) => Some(Expr::Const(Const::Int(i)).typed(Type::Int)),
                Token::LongLiteral(i) => Some(Expr::Const(Const::Long(i)).typed(Type::Long)),
                _ => None,
            }
        }),
        // group
        delimited(tag_token!(Token::ParenOpen), parse_expr, tag_token!(Token::ParenClose)),
        // variable
        parse_ident.map(Expr::Var).map(Expr::dummy_typed),
    ))
    .parse_complete(i)
}

fn parse_expr(i: Tokens<'_>) -> IResult<Tokens<'_>, TypedExpr, ParseError<'_>> {
    // precedence reference:
    // https://en.cppreference.com/w/c/language/operator_precedence

    precedence(
        parse_prefixes,
        parse_postfixes,
        parse_infixes,
        parse_operands,
        // fold
        |op| -> Result<TypedExpr, ()> {
            use Operation::*;
            Ok(match op {
                Prefix(Left(op), exp) | Postfix(exp, op) => Expr::Unary(op, Box::new(exp)),
                Prefix(Right(ty), exp) => Expr::Cast { target: ty, inner: Box::new(exp) },
                Binary(lhs, Typical(op), rhs) => {
                    Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
                }
                Binary(lhs, CompoundAssignment(op), rhs) => {
                    Expr::CompoundAssignment { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
                }
                Binary(lhs, Assignment, rhs) => Expr::Assignemnt(Box::new(lhs), Box::new(rhs)),
                Binary(lhs, Ternary(op), rhs) => Expr::Conditional {
                    cond: Box::new(lhs),
                    then: Box::new(op),
                    else_: Box::new(rhs),
                },
            }
            // apparently it is a bad idea to infer types here.
            .dummy_typed())
        },
    )(i)
}

use std::clone::Clone;

use crate::{
    ast::*,
    lexer::{IntFlags, Token, Tokens},
};
use ecow::EcoString as Ecow;
use either::Either::{self, Left, Right};
use nom::{
    Finish, Parser,
    branch::alt,
    bytes::take,
    combinator::{all_consuming, fail, opt, success, verify},
    error,
    multi::{fold, many, separated_list0, separated_list1},
    sequence::{delimited, preceded, separated_pair, terminated},
};
use nom_language::precedence::{Assoc, Binary, Operation, Unary, binary_op, precedence, unary_op};

// type ParseError<'s> = ();
type ParseError<'s> = (Tokens<'s>, nom::error::ErrorKind);
type ParseResult<'s, T> = nom::IResult<Tokens<'s>, T, ParseError<'s>>;

pub fn parse(tokens: &[Token]) -> Result<Program, ParseError<'_>> {
    all_consuming(parse_program).parse_complete(Tokens(tokens)).finish().map(|t| t.1)
}

fn parse_program(i: Tokens<'_>) -> ParseResult<'_, Program> {
    many(1.., parse_decl).map(Program::new).parse_complete(i)
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

fn parse_var_decl(i: Tokens<'_>) -> ParseResult<'_, VarDecl> {
    parse_decl
        .map_opt(|decl| match decl {
            Decl::Func(_) => None,
            Decl::Var(var) => Some(var),
        })
        .parse_complete(i)
}

fn parse_decl(i: Tokens<'_>) -> ParseResult<'_, Decl> {
    let (i, (ret, sc)) = parse_specifiers.parse_complete(i)?;

    let (i, (name, decl_type, param_names)) = parse_declarator
        .map_opt(|declarator| process_declarator(declarator, ret.clone()))
        .parse_complete(i)?;

    let result = match decl_type {
        Type::Func { .. } => {
            let (i, body) =
                parse_block.map(Some).or(tag_token!(Token::Semicolon => None)).parse_complete(i)?;

            (i, Decl::Func(FuncDecl { name, params: param_names, body, sc, fun_type: decl_type }))
        }
        _ => {
            let (i, init) = terminated(
                opt(preceded(tag_token!(Token::Equal), parse_expr)),
                tag_token!(Token::Semicolon),
            )
            .parse_complete(i)?;

            (i, Decl::Var(VarDecl { name, init, sc, var_type: decl_type }))
        }
    };

    Ok(result)
}

enum Declarator {
    Ident(Ecow),
    Pointer(Box<Declarator>),
    Func(Vec<ParamInfo>, Box<Declarator>),
}

struct ParamInfo {
    type_: Type,
    declarator: Declarator,
}

fn parse_param_info(i: Tokens<'_>) -> ParseResult<'_, Vec<ParamInfo>> {
    delimited(
        tag_token!(Token::ParenOpen),
        alt((
            tag_token!(Token::Void => Vec::new()),
            separated_list1(
                tag_token!(Token::Comma),
                parse_base_type
                    .and(parse_declarator)
                    .map(|(type_, declarator)| ParamInfo { type_, declarator }),
            ),
        )),
        tag_token!(Token::ParenClose),
    )
    .parse_complete(i)
}

fn parse_declarator(i: Tokens<'_>) -> ParseResult<'_, Declarator> {
    precedence(
        // prefix: pointer
        unary_op(2, tag_token!(Token::Astrisk => ())),
        // postfix : function
        unary_op(1, parse_param_info),
        // infix, inapplicable
        fail(),
        // operands
        alt((
            delimited(
                tag_token!(Token::ParenOpen),
                parse_declarator,
                tag_token!(Token::ParenClose),
            ),
            parse_ident.map(Declarator::Ident),
        )),
        // fold function
        |op: Operation<_, _, (), _>| -> Result<Declarator, ()> {
            let ret = match op {
                Operation::Prefix((), d) => Declarator::Pointer(Box::new(d)),
                Operation::Postfix(d, params) => Declarator::Func(params, Box::new(d)),
                Operation::Binary(_, _, _) => return Err(()),
            };
            Ok(ret)
        },
    )(i)
}

fn process_declarator(declarator: Declarator, base: Type) -> Option<(Ecow, Type, Vec<Ecow>)> {
    match declarator {
        Declarator::Ident(name) => Some((name, base, Vec::new())),
        Declarator::Pointer(declarator) => {
            let derived = Type::Pointer { to: Box::new(base) };
            process_declarator(*declarator, derived)
        }
        Declarator::Func(params, d) => match *d {
            Declarator::Ident(name) => {
                let mut param_types = Vec::with_capacity(params.len());
                let mut param_names = Vec::with_capacity(params.len());
                for param in params {
                    let (pname, pty, _) =
                        process_declarator(param.declarator, param.type_.clone())?;

                    if matches!(param.type_, Type::Func { .. }) {
                        // should coerce to a function pointer
                        return None;
                    }

                    param_types.push(pty);
                    param_names.push(pname);
                }

                let derived = Type::Func { params: param_types, ret: Box::new(base) };

                Some((name, derived, param_names))
            }
            // function pointers can be added starting here
            _ => None,
        },
    }
}

fn parse_base_type(i: Tokens<'_>) -> ParseResult<'_, Type> {
    // i really hate this function.

    many(
        1..,
        tag_token!(Token::Int | Token::Long | Token::Double | Token::Signed | Token::Unsigned)
            .map(|t: Tokens<'_>| t.0[0].clone()),
    )
    .map_opt(|list: Vec<_>| {
        use Token::*;

        if list == vec![Double] {
            return Some(Type::Double);
        } else if list.contains(&Double) {
            return None;
        }

        let has_dupes = 'dupes: {
            for i in 0..list.len() {
                for j in i + 1..list.len() {
                    if list[i] == list[j] {
                        break 'dupes true;
                    }
                }
            }
            false
        };

        if list.is_empty() || has_dupes || (list.contains(&Signed) && list.contains(&Unsigned)) {
            return None;
        }

        let u = list.contains(&Unsigned);
        let l = list.contains(&Long);

        Some(match (u, l) {
            (true, true) => Type::ULong,
            (true, false) => Type::UInt,
            (false, true) => Type::Long,
            (false, false) => Type::Int,
        })
    })
    .parse_complete(i)
}

fn parse_specifiers(i: Tokens<'_>) -> ParseResult<'_, (Type, StorageClass)> {
    let (i, (sc, ty)) = fold(
        1..,
        alt((
            tag_token!(Token::Static => StorageClass::Static).map(Left),
            tag_token!(Token::Extern => StorageClass::Extern).map(Left),
            tag_token!(Token::Long => Token::Long).map(Right),
            tag_token!(Token::Int => Token::Int).map(Right),
            tag_token!(Token::Signed => Token::Signed).map(Right),
            tag_token!(Token::Unsigned => Token::Unsigned).map(Right),
            tag_token!(Token::Double => Token::Double).map(Right),
        )),
        || (Vec::new(), Vec::new()),
        |(mut sc, mut types), item| {
            match item {
                Left(s) => sc.push(s),
                Right(t) => types.push(t),
            }

            (sc, types)
        },
    )
    .parse_complete(i)?;

    if sc.len() > 1 {
        return Err(nom::Err::Error(error::make_error(i, error::ErrorKind::TooLarge)));
    }

    let sc = sc.first().map_or(StorageClass::None, |i| *i);
    let (_, ty) = parse_base_type(Tokens::from(&ty[..])).map_err(|e| e.map_input(|_| i))?;

    Ok((i, (ty, sc)))
}

fn parse_block(i: Tokens<'_>) -> ParseResult<'_, Block> {
    delimited(
        tag_token!(Token::BraceOpen),
        many(.., parse_decl.map(BlockItem::D).or(parse_stmt.map(BlockItem::S))),
        tag_token!(Token::BraceClose),
    )
    .map(Block)
    .parse_complete(i)
}

fn parse_ident(i: Tokens<'_>) -> ParseResult<'_, Ecow> {
    tag_token!(Token::Ident(_)).map_opt(|t: Tokens<'_>| t.0[0].unwrap_ident()).parse_complete(i)
}

fn parse_stmt(i: Tokens<'_>) -> ParseResult<'_, Stmt> {
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

#[derive(Debug, Clone)]
enum AbstractDeclarator {
    Pointer(Box<AbstractDeclarator>),
    Base,
}

fn parse_abstract_declarator(i: Tokens<'_>) -> ParseResult<'_, AbstractDeclarator> {
    precedence(
        // pointer
        unary_op(2, tag_token!(Token::Astrisk => ())),
        fail(),
        fail(),
        alt((
            delimited(
                tag_token!(Token::ParenOpen),
                parse_abstract_declarator,
                tag_token!(Token::ParenClose),
            ),
            success(AbstractDeclarator::Base),
        )),
        |op: Operation<_, (), (), _>| -> Result<AbstractDeclarator, ()> {
            match op {
                Operation::Prefix((), d) => Ok(AbstractDeclarator::Pointer(Box::new(d))),
                Operation::Postfix(_, _) | Operation::Binary(_, _, _) => Err(()),
            }
        },
    )(i)
}

fn process_abstract_declarator(declarator: AbstractDeclarator, base: Type) -> Option<Type> {
    match declarator {
        AbstractDeclarator::Pointer(declarator) => {
            let derived = Type::Pointer { to: Box::new(base) };
            process_abstract_declarator(*declarator, derived)
        }
        AbstractDeclarator::Base => Some(base),
    }
}

fn parse_prefixes(i: Tokens<'_>) -> ParseResult<'_, Unary<Either<UnaryOp, Type>, i32>> {
    alt((
        unary_op(2, tag_token!(Token::Hyphen => Left(UnaryOp::Negate))),
        unary_op(2, tag_token!(Token::Tilde => Left(UnaryOp::Complement))),
        unary_op(2, tag_token!(Token::Bang => Left(UnaryOp::Not))),
        unary_op(2, tag_token!(Token::Plus => Left(UnaryOp::Plus))),
        // chapter 5 extra credit
        unary_op(2, tag_token!(Token::DblPlus => Left(UnaryOp::IncPre))),
        unary_op(2, tag_token!(Token::DblHyphen => Left(UnaryOp::DecPre))),
        // chapter 11 cast operation // also chapter 14 declarators (todo)
        unary_op(
            2,
            delimited(
                tag_token!(Token::ParenOpen),
                parse_base_type
                    .and(parse_abstract_declarator)
                    .map_opt(|(ty, de)| process_abstract_declarator(de, ty)),
                tag_token!(Token::ParenClose),
            )
            .map(Right),
        ),
        // chapter 14 pointers // is this enough?
        unary_op(2, tag_token!(Token::Ambersand => Left(UnaryOp::AddressOf))),
        unary_op(2, tag_token!(Token::Astrisk => Left(UnaryOp::Dereference))),
    ))
    .parse_complete(i)
}

fn parse_postfixes(i: Tokens<'_>) -> ParseResult<'_, Unary<UnaryOp, i32>> {
    // chapter 5 extra credit

    alt((
        unary_op(1, tag_token!(Token::DblPlus => UnaryOp::IncPost)),
        unary_op(1, tag_token!(Token::DblHyphen => UnaryOp::DecPost)),
    ))
    .parse_complete(i)
}

fn parse_infixes(i: Tokens<'_>) -> ParseResult<'_, Binary<BinKind, i32>> {
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

macro_rules! typed_const {
    ($ty:tt, $i:expr) => {
        Expr::Const(Const::$ty($i)).typed(Type::$ty)
    };
}

fn parse_int_literal(i: Tokens<'_>) -> ParseResult<'_, TypedExpr> {
    tag_token!(Token::Integer(..))
        .map(|t: Tokens<'_>| {
            let Token::Integer(lit, ty) = t.0[0] else { unreachable!() };

            match ty {
                IntFlags::U => match u32::try_from(lit) {
                    Ok(i) => typed_const!(UInt, i),
                    Err(_) => typed_const!(ULong, lit),
                },
                IntFlags::UL => typed_const!(ULong, lit),
                IntFlags::L => typed_const!(Long, lit as i64),
                IntFlags::NoFlags => match i32::try_from(lit) {
                    Ok(i) => typed_const!(Int, i),
                    Err(_) => typed_const!(Long, lit as i64),
                },
            }
        })
        .parse_complete(i)
}

fn parse_operands(i: Tokens<'_>) -> ParseResult<'_, TypedExpr> {
    alt((
        // const int
        parse_int_literal,
        // const float
        tag_token!(Token::Float(_)).map(|tokens: Tokens<'_>| {
            let Token::Float(cnst) = &tokens.0[0] else { unreachable!() };
            typed_const!(Double, *cnst)
        }),
        // function call
        parse_ident
            .and(delimited(
                tag_token!(Token::ParenOpen),
                separated_list0(tag_token!(Token::Comma), parse_expr),
                tag_token!(Token::ParenClose),
            ))
            .map(|(name, args)| Expr::FuncCall { name, args }.dummy_typed()),
        // group
        delimited(tag_token!(Token::ParenOpen), parse_expr, tag_token!(Token::ParenClose)),
        // variable
        parse_ident.map(Expr::Var).map(Expr::dummy_typed),
    ))
    .parse_complete(i)
}

fn parse_expr(i: Tokens<'_>) -> ParseResult<'_, TypedExpr> {
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
                Prefix(Left(UnaryOp::AddressOf), exp) => Expr::AddrOf(Box::new(exp)),
                Prefix(Left(UnaryOp::Dereference), exp) => Expr::Deref(Box::new(exp)),
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

use std::clone::Clone;

use crate::{
    ast::*,
    lexer::{IntFlags, Token, Tokens},
};
use ecow::EcoString as Identifier;
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

type ParseError<'s> = (Tokens<'s>, error::ErrorKind);
// type ParseError<'s> = nom_language::error::VerboseError<Tokens<'s>>;
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
            take((tag_token!(@c $($token),+))),
            |t: &Tokens| matches!(t.0[0..tag_token!(@c $($token),+)],[$($token),+])
        )
        .context(concat!("token_tag: ", stringify!($($token),+)))
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
        .context("variable declaration")
        .parse_complete(i)
}

fn parse_decl(i: Tokens<'_>) -> ParseResult<'_, Decl> {
    let (i, (base_type, sc)) =
        parse_specifiers.context("specifiers in declaration").parse_complete(i)?;

    let (i, (name, decl_type, params)) = parse_declarator
        .map_opt(|d| process_declarator(d, base_type.clone()))
        .context("declarator in declaration")
        .parse_complete(i)?;

    let result = match decl_type {
        Type::Func { .. } => {
            let (i, body) = parse_block
                .map(Some)
                .or(tag_token!(Token::Semicolon => None))
                .context("functiom body declaration")
                .parse_complete(i)?;

            (i, Decl::Func(FuncDecl { name, params, body, sc, fun_type: decl_type }))
        }
        _ => {
            let (i, init) = terminated(
                opt(preceded(
                    tag_token!(Token::Equal),
                    parse_initializaer.context("variable init declaration"),
                )),
                tag_token!(Token::Semicolon),
            )
            .parse_complete(i)?;

            (i, Decl::Var(VarDecl { name, init, sc, var_type: decl_type }))
        }
    };

    Ok(result)
}

fn parse_initializaer(i: Tokens<'_>) -> ParseResult<'_, Initializer> {
    alt((
        parse_expr.map(Initializer::Single),
        delimited(
            tag_token!(Token::BraceOpen),
            terminated(
                separated_list1(tag_token!(Token::Comma), parse_initializaer),
                opt(tag_token!(Token::Comma)),
            ),
            tag_token!(Token::BraceClose),
        )
        .map(Initializer::Compound),
    ))
    .context("initializer")
    .parse_complete(i)
}

#[derive(Debug)]
enum Declarator {
    Ident(Identifier),
    Pointer(Box<Declarator>),
    Array(Box<Declarator>, usize),
    Func(Vec<ParamInfo>, Box<Declarator>),
}

#[derive(Debug)]
struct ParamInfo {
    type_: Type,
    declarator: Declarator,
}

fn parse_param_info(i: Tokens<'_>) -> ParseResult<'_, Vec<ParamInfo>> {
    parenthesized(alt((
        tag_token!(Token::Void => Vec::new()),
        separated_list1(
            tag_token!(Token::Comma),
            parse_base_type
                .and(parse_declarator)
                .map(|(type_, declarator)| ParamInfo { type_, declarator }),
        ),
    )))
    .context("paramerer info")
    .parse_complete(i)
}

fn parse_declarator(i: Tokens<'_>) -> ParseResult<'_, Declarator> {
    precedence(
        unary_op(2, tag_token!(Token::Astrisk => ())),
        alt((
            // function
            unary_op(1, parse_param_info.map(Right)),
            // array
            unary_op(
                1,
                bracketed(tag_token!(Token::Integer { .. }).map_opt(|t: Tokens<'_>| {
                    match t.0[0] {
                        Token::Integer(0, _) => None,
                        Token::Integer(i, _) => Some(i as usize),
                        _ => unreachable!(),
                    }
                }))
                .map(Left),
            ),
        )),
        fail(),
        parenthesized(parse_declarator).or(parse_ident.map(Declarator::Ident)),
        |op| {
            let ret = match op {
                Operation::Prefix((), d) => Declarator::Pointer(Box::new(d)),
                Operation::Postfix(d, Right(params)) => Declarator::Func(params, Box::new(d)),
                Operation::Postfix(d, Left(size)) => Declarator::Array(Box::new(d), size),
                Operation::Binary(_, (), _) => return Err(()),
            };
            Ok(ret)
        },
    )
    .parse_complete(i)
}

fn process_declarator(
    declarator: Declarator,
    base: Type,
) -> Option<(Identifier, Type, Vec<Identifier>)> {
    match declarator {
        Declarator::Ident(name) => Some((name, base, Vec::new())),
        Declarator::Pointer(declarator) => {
            let derived = Type::Pointer { to: Box::new(base) };
            process_declarator(*declarator, derived)
        }
        Declarator::Array(declarator, size) => {
            let derived = Type::Array { element: Box::new(base), size };
            process_declarator(*declarator, derived)
        }
        Declarator::Func(..) if matches!(base, Type::Array { .. }) => None,
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

        let has_dupes = (0..list.len())
            .flat_map(|i| (i + 1..list.len()).map(move |j| (i, j)))
            .any(|(i, j)| list[i] == list[j]);

        let unsigned = list.contains(&Unsigned);

        if list == vec![Double] {
            Some(Type::Double)
        } else if list.contains(&Double)
            || list.is_empty()
            || has_dupes
            || (list.contains(&Signed) && unsigned)
        {
            None
        } else {
            Some(match (unsigned, list.contains(&Long)) {
                (true, true) => Type::ULong,
                (true, false) => Type::UInt,
                (false, true) => Type::Long,
                (false, false) => Type::Int,
            })
        }
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
    let (_, ty) = parse_base_type(Tokens(&ty[..]))
        .map_err(|e| e.map(|_| nom::error::make_error(i, error::ErrorKind::CrLf)))?;

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

fn parse_ident(i: Tokens<'_>) -> ParseResult<'_, Identifier> {
    tag_token!(Token::Ident(_)).map_opt(|t: Tokens<'_>| t.0[0].unwrap_ident()).parse_complete(i)
}

fn parse_stmt(i: Tokens<'_>) -> ParseResult<'_, Stmt> {
    let ret = delimited(tag_token!(Token::Return), parse_expr, tag_token!(Token::Semicolon))
        .map(Stmt::Return);
    let expr = terminated(parse_expr, tag_token!(Token::Semicolon)).map(Stmt::Expression);

    let if_else = preceded(
        tag_token!(Token::If),
        (
            parenthesized(parse_expr),
            parse_stmt.map(Box::new),
            opt(preceded(tag_token!(Token::Else), parse_stmt).map(Box::new)),
        ),
    )
    .map(|(cond, then, else_)| Stmt::If { cond, then, else_ });

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
    .context("statement")
    .parse_complete(i)
}

#[derive(Debug, Clone)]
enum AbsDeclarator {
    Pointer(Box<AbsDeclarator>),
    Array(Box<AbsDeclarator>, usize),
    Base,
}

fn parse_abstract_declarator(i: Tokens<'_>) -> ParseResult<'_, AbsDeclarator> {
    precedence(
        unary_op(2, tag_token!(Token::Astrisk => ())),
        unary_op(
            1,
            bracketed(tag_token!(Token::Integer { .. }).map_opt(|t: Tokens<'_>| match t.0[0] {
                Token::Integer(0, _) => None,
                Token::Integer(i, _) => Some(i as usize),
                _ => unreachable!(),
            })),
        ),
        fail(),
        parenthesized(parse_abstract_declarator).or(success(AbsDeclarator::Base)),
        |op| match op {
            Operation::Prefix((), d) => Ok(AbsDeclarator::Pointer(Box::new(d))),
            Operation::Postfix(d, size) => Ok(AbsDeclarator::Array(Box::new(d), size)),
            Operation::Binary(_, (), _) => Err(()),
        },
    )
    .parse_complete(i)
}

fn process_abstract_declarator(declarator: AbsDeclarator, base: Type) -> Type {
    match declarator {
        AbsDeclarator::Pointer(declarator) => {
            let derived = Type::Pointer { to: Box::new(base) };
            process_abstract_declarator(*declarator, derived)
        }
        AbsDeclarator::Array(declarator, size) => {
            let derived = Type::Array { element: Box::new(base), size };
            process_abstract_declarator(*declarator, derived)
        }
        AbsDeclarator::Base => base,
    }
}

fn parse_prefixes(i: Tokens<'_>) -> ParseResult<'_, Unary<Either<UnaryOp, Type>, i32>> {
    alt((
        unary_op(2, tag_token!(Token::Hyphen => UnaryOp::Negate).map(Left)),
        unary_op(2, tag_token!(Token::Tilde => UnaryOp::Complement).map(Left)),
        unary_op(2, tag_token!(Token::Bang => UnaryOp::Not).map(Left)),
        unary_op(2, tag_token!(Token::Plus => UnaryOp::Plus).map(Left)),
        // chapter 5 extra credit
        unary_op(2, tag_token!(Token::DblPlus => UnaryOp::IncPre).map(Left)),
        unary_op(2, tag_token!(Token::DblHyphen => UnaryOp::DecPre).map(Left)),
        // chapter 11 cast operation // also chapter 14 declarators
        unary_op(
            2,
            parenthesized(
                parse_base_type
                    .and(parse_abstract_declarator)
                    .map(|(ty, de)| process_abstract_declarator(de, ty)),
            )
            .map(Right),
        ),
        // chapter 14 pointers
        unary_op(2, tag_token!(Token::Ambersand => UnaryOp::AddressOf).map(Left)),
        unary_op(2, tag_token!(Token::Astrisk => UnaryOp::Dereference).map(Left)),
    ))
    .parse_complete(i)
}

fn parse_postfixes(i: Tokens<'_>) -> ParseResult<'_, Unary<Either<UnaryOp, TypedExpr>, i32>> {
    alt((
        // chapter 5 extra credit
        unary_op(1, tag_token!(Token::DblPlus => UnaryOp::IncPost).map(Left)),
        unary_op(1, tag_token!(Token::DblHyphen => UnaryOp::DecPost).map(Left)),
        // chapter 15
        unary_op(1, bracketed(parse_expr).map(Right)),
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

fn parse_int_literal(i: Tokens<'_>) -> ParseResult<'_, TypedExpr> {
    tag_token!(Token::Integer(..))
        .map(|t: Tokens<'_>| {
            let Token::Integer(lit, ty) = t.0[0] else { unreachable!() };

            let c = match ty {
                IntFlags::U => match u32::try_from(lit) {
                    Ok(i) => Const::UInt(i),
                    Err(_) => Const::ULong(lit),
                },
                IntFlags::UL => Const::ULong(lit),
                IntFlags::L => Const::Long(lit as i64),
                IntFlags::NoFlags => match i32::try_from(lit) {
                    Ok(i) => Const::Int(i),
                    Err(_) => Const::Long(lit as i64),
                },
            };

            Expr::Const(c).dummy_typed()
        })
        .parse_complete(i)
}

fn parse_operands(i: Tokens<'_>) -> ParseResult<'_, TypedExpr> {
    alt((
        // const int
        parse_int_literal.context("integer literal"),
        // const float
        tag_token!(Token::Float(_))
            .map(|tokens: Tokens<'_>| {
                let Token::Float(cnst) = &tokens.0[0] else { unreachable!() };
                Expr::Const(Const::Double(*cnst)).typed(Type::Double)
            })
            .context("float literal"),
        // function call
        parse_ident
            .and(parenthesized(separated_list0(tag_token!(Token::Comma), parse_expr)))
            .map(|(name, args)| Expr::FuncCall { name, args }.dummy_typed())
            .context("function call expr"),
        // group
        parenthesized(parse_expr),
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
                Prefix(Left(op), exp) | Postfix(exp, Left(op)) => Expr::Unary(op, Box::new(exp)),
                Postfix(e1, Right(e2)) => Expr::Subscript(Box::new(e1), Box::new(e2)),
                Prefix(Right(ty), exp) => Expr::Cast { target: ty, inner: Box::new(exp) },
                Binary(lhs, Typical(op), rhs) => {
                    Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
                }
                Binary(lhs, CompoundAssignment(op), rhs) => Expr::CompoundAssignment {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    common: None,
                },
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
    )
    .context("expression")
    .parse_complete(i)
}

fn parenthesized<'a, P, O>(parser: P) -> impl Parser<Tokens<'a>, Output = O, Error = ParseError<'a>>
where
    P: Parser<Tokens<'a>, Output = O, Error = ParseError<'a>>,
{
    delimited(tag_token!(Token::ParenOpen), parser, tag_token!(Token::ParenClose))
}

fn bracketed<'a, P, O>(parser: P) -> impl Parser<Tokens<'a>, Output = O, Error = ParseError<'a>>
where
    P: Parser<Tokens<'a>, Output = O, Error = ParseError<'a>>,
{
    delimited(tag_token!(Token::BracketOpen), parser, tag_token!(Token::BracketClose))
}

// helper trait for diagnostics
trait InnerContext
where
    Self: std::marker::Sized,
{
    fn context(self, msg: &'static str) -> error::Context<Self> {
        error::context(msg, self)
    }
}
impl<'s, T> InnerContext for T where T: Parser<Tokens<'s>> {}

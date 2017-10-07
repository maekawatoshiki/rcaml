use nom::{IResult, alpha, alphanumeric, digit, double, space};

use std::str;
use std::str::FromStr;
use std::collections::HashMap;

use node;
use node::NodeKind;

use typing::Type;

use std::boxed::Box;

// syntax reference: https://caml.inria.fr/pub/docs/manual-ocaml/language.html

fn to_str(slice: &[u8]) -> &str {
    str::from_utf8(slice).unwrap()
}

named!(comment<()>, do_parse!(
    tag!("(*") >>
    take_until!("*)") >> ()
));

named!(whitespace<()>, do_parse!(
    one_of!(" \t\n\r") >> ()
));

named!(opt_spaces<()>, do_parse!(
    many0!(alt!(whitespace | comment)) >> ()
));

named!(spaces<()>, do_parse!(
    many1!(alt!(whitespace | comment)) >> ()
));

named!(funcdef<NodeKind>, 
    do_parse!(
        name:   ident >> // TODO: not only identifier... (https://caml.inria.fr/pub/docs/manual-ocaml/patterns.html#pattern)
        params: many1!(do_parse!(spaces >> param: ident >> (param))) >>
        (NodeKind::FuncDef(
                (name.get_ident_name().unwrap(), Type::Var(0)), 
                (params.into_iter().map(|param| ( param.get_ident_name().unwrap(), Type::Var(0) ) ).collect())
                )
        )
    )
);

named!(expr_let<NodeKind>,
    do_parse!(
        tag!("let") >>
        spaces >> 
        name: alt!(funcdef | ident) >> 
        ws!(tag!("=")) >>
        exp: expr >>
        spaces >>
        tag!("in") >> 
        spaces >> 
        body: expr >>
        (match name {
            NodeKind::FuncDef(name, params) => NodeKind::LetFuncExpr(
                                                node::FuncDef { name: name, params: params },
                                                Box::new(exp),
                                                Box::new(body)
                                               ),
            NodeKind::Ident(name)           => NodeKind::LetExpr(
                                                (name, Type::Var(0)),
                                                Box::new(exp),
                                                Box::new(body)
                                               ),
            _                               => panic!()
        })
    )
);

named!(expr<NodeKind>, 
    alt!(
            expr_let
        |   expr_add_sub
    )
);

named!(expr_mul_div<NodeKind>,
    do_parse!(
        init: expr_unary >> 
        res:  fold_many0!(
                do_parse!(
                    opt_spaces >> 
                    op: alt!(tag!("*.") | tag!("/.") | tag!("*") | tag!("/")) >> 
                    opt_spaces >> 
                    rhs: expr_mul_div >> 
                    (op, rhs)
                ),
                init,
                |n1, (op, n2): (&[u8], NodeKind)| {
                    let (op, is_int) = node::str_to_binop(str::from_utf8(op).unwrap());
                    if is_int { NodeKind::IntBinaryOp(op, Box::new(n1), Box::new(n2)) } 
                    else { NodeKind::FloatBinaryOp(op, Box::new(n1), Box::new(n2)) }
                }
        ) >> (res)
    )
);

named!(expr_add_sub<NodeKind>,
    do_parse!(
        init: expr_mul_div >> 
        res:  fold_many0!(
                do_parse!(
                    opt_spaces >> 
                    op: alt!(tag!("+.") | tag!("-.") | tag!("+") | tag!("-")) >> 
                    opt_spaces >> 
                    rhs: expr_mul_div >> 
                    (op, rhs)
                ),
                init,
                |n1, (op, n2): (&[u8], NodeKind)| {
                    let (op, is_int) = node::str_to_binop(str::from_utf8(op).unwrap());
                    if is_int { NodeKind::IntBinaryOp(op, Box::new(n1), Box::new(n2)) } 
                    else { NodeKind::FloatBinaryOp(op, Box::new(n1), Box::new(n2)) }
                }
        ) >> (res)
    )
);

named!(expr_func_call<NodeKind>,
    do_parse!(
        p: expr_prim >>
        args:   fold_many0!(
                expr,
                Vec::new(),
                |mut a: Vec<NodeKind>, arg: NodeKind| {
                    a.push(arg);
                    a
                }
        ) >> (if args.len() == 0 { p } else { NodeKind::Call(Box::new(p), args) })
    )
);

named!(expr_unary<NodeKind>,
    alt!(
        do_parse!(
            opt_spaces >> 
            op: alt!(tag!("-.") | tag!("-")) >> 
            opt_spaces >> 
            e: expr_unary >> 
            (NodeKind::UnaryOp(node::str_to_unaryop(str::from_utf8(op).unwrap()), Box::new(e)))
        ) | 
        expr_postfix
    )
);

named!(apply_postfix<Vec<NodeKind>>, do_parse!(
    spaces >>
    args: separated_nonempty_list_complete!(spaces, expr_prim) >>
    (args)
));

named!(expr_postfix<NodeKind>,
    do_parse!(
        init: expr_prim >> 
        folded: fold_many0!(
            apply_postfix,
            init,
            |lhs, pf| {
                NodeKind::Call(Box::new(lhs), pf)
            }
        ) >> (folded)
    ) 
);

named!(expr_prim<NodeKind>,
    alt!(
          constant 
        | parens
    )
);

named!(integer<NodeKind>, 
    do_parse!(
        i: map_res!(map_res!(
            digit,
            str::from_utf8
        ), FromStr::from_str) >>
        (NodeKind::Int(i))
    )
);

named!(float<NodeKind>,
    do_parse!(
        f: double >> 
        (NodeKind::Float(f))
    )
);

fn is_ident(x: &[u8]) -> bool {
    let keywords = vec![&b"let"[..], &b"rec"[..], &b"in"[..], &b"true"[..],
                        &b"false"[..], &b"if"[..], &b"then"[..], &b"else"[..],
                        &b"Array.create"[..], &b"Array.make"[..]];
    if x.len() == 0 || keywords.contains(&x) {
        return false;
    }
    !(b'0' <= x[0] && x[0] <= b'9')
}
fn is_not_ident_u8(x: u8) -> bool {
    !((b'0' <= x && x <= b'9') || (b'A' <= x && x <= b'Z') || (b'a' <= x && x <= b'z') || x == b'_')
}

named!(ident<NodeKind>, do_parse!(
    i: verify!(take_till!(is_not_ident_u8), is_ident) >>
    (NodeKind::Ident(String::from_utf8(i.to_vec()).unwrap()))
));

named!(bool_true<NodeKind>,
    do_parse!( tag!("true") >> (NodeKind::Bool(true)) )
);

named!(bool_false<NodeKind>,
    do_parse!( tag!("false") >> (NodeKind::Bool(false)) )
);

named!(constant<NodeKind>,
    alt_complete!(float | integer | ident | bool_false | bool_true)
);

named!(parens<NodeKind>, delimited!(tag!("("), expr, tag!(")")));


named!(opt_dscolon<()>, do_parse!(
    many0!(tag!(";;")) >> ()
));

named!(module_item<NodeKind>,
    do_parse!(
        ws!(opt_dscolon) >> 
        i: alt!(expr | definition) >> 
        opt_spaces >> 
        opt_dscolon >> (i)
    )
);

named!(definition<NodeKind>,
    alt!(
        definition_let 
    )
);

named!(definition_let<NodeKind>,
    do_parse!(
        tag!("let") >>
        spaces >> 
        name: alt!(funcdef | ident) >> 
        ws!(tag!("=")) >>
        exp: expr >>
        (match name {
            NodeKind::FuncDef(name, params) => NodeKind::LetFuncDef(
                                                node::FuncDef { name: name, params: params },
                                                Box::new(exp)),
            NodeKind::Ident(name)           => NodeKind::LetDef(
                                                (name, Type::Var(0)),
                                                Box::new(exp)
                                               ),
            _                               => panic!()
        })
    )
);

pub fn parse_simple_expr(e: &str) {
    println!("expr: {}\n{}", e, match expr(e.as_bytes()) {
        IResult::Done(_, expr_node) => format!("generated node: {:?}", expr_node),
        IResult::Incomplete(needed) => format!("imcomplete: {:?}",     needed),
        IResult::Error(err) =>         format!("error: {:?}",          err)
    });
}

pub fn parse_module_item_expr(e: &str) {

    println!("module-item: {}\n{}", e, match module_item(e.as_bytes()) {
        IResult::Done(_, expr_node) => format!("generated node: {:?}", expr_node),
        IResult::Incomplete(needed) => format!("imcomplete: {:?}",     needed),
        IResult::Error(err) =>         format!("error: {:?}",          err)
    });

    // sloppy impl of showing type-infered node
    use typing;
    use id;
    match module_item(e.as_bytes()) {
        IResult::Done(_, expr_node) => {
            let mut idgen = id::IdGen::new();
            println!("type infered node: {:?}", typing::f(&expr_node, &mut idgen));
        }
        _ => panic!(),
    }
}

#[test]
pub fn test_parse_simple_expr() {
    use node::NodeKind::*;
    use node::BinOps::*;

    let f = |e: &str| match expr(e.as_bytes()) {
        IResult::Done(_, expr_node) => expr_node,
        IResult::Incomplete(needed) => panic!(format!("imcomplete: {:?}",     needed)),
        IResult::Error(err) => panic!(format!("error: {:?}", err)),
    };

    assert_eq!(f("5 / a3 + 11 * 10"),
               IntBinaryOp(IAdd,
                        Box::new(IntBinaryOp(IDiv,
                                          Box::new(Int(5)),
                                          Box::new(Ident("a3".to_string())))),
                        Box::new(IntBinaryOp(IMul,
                                          Box::new(Int(11)),
                                          Box::new(Int(10))))));
    assert_eq!(f("5.3 *. 10.2"),
               FloatBinaryOp(FMul,
                        Box::new(Float(5.3)),
                        Box::new(Float(10.2))))
}

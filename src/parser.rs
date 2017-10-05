use nom::{IResult, alpha, alphanumeric, digit, double};

use std::str;
use std::str::FromStr;

use node::{NodeKind, BinOps};
use node;
use std::boxed::Box;

// syntax reference: https://caml.inria.fr/pub/docs/manual-ocaml/language.html

fn to_str(slice: &[u8]) -> &str {
    str::from_utf8(slice).unwrap()
}

named!(expr<NodeKind>, 
    do_parse!(
        e: expr_add_sub >> 
        (e)
    )
);

named!(expr_mul_div<NodeKind>,
    do_parse!(
        init: expr_prim >> 
        res:  fold_many0!(
                pair!(alt!(
                        tag!("*.") |
                        tag!("/.") |
                        tag!("*" ) |
                        tag!("/" ) 
                            ), expr_prim),
                init,
                |n1, (op, n2): (&[u8], NodeKind)| {
                    let op = node::str_to_binop(str::from_utf8(op).unwrap());
                    NodeKind::BinaryOp(op, Box::new(n1), Box::new(n2))
                }
        ) >>
        (res)
    )
);

named!(expr_add_sub<NodeKind>,
    do_parse!(
        init: expr_mul_div >> 
        res:  fold_many0!(
                pair!(alt!(
                        tag!("+.") |
                        tag!("-.") |
                        tag!("+" ) |
                        tag!("-" ) 
                            ), expr_mul_div),
                init,
                |n1, (op, n2): (&[u8], NodeKind)| {
                    let op = node::str_to_binop(str::from_utf8(op).unwrap());
                    NodeKind::BinaryOp(op, Box::new(n1), Box::new(n2))
                }
        ) >>
        (res)
    )
);

named!(integer<NodeKind>, 
    do_parse!(
        i: map_res!(map_res!(
            ws!(digit),
            str::from_utf8
        ), FromStr::from_str) >>
        (NodeKind::Int(i))
    )
);

named!(float<NodeKind>,
    do_parse!(
        f: ws!(double) >> 
        (NodeKind::Float(f))
    )
);

named!(ident<NodeKind>,
    ws!(do_parse!(
        bgn:    alt!(alpha | tag!("_")) >> 
        remain: opt!(alphanumeric) >> 
        (NodeKind::Ident(
            if let Some(s) = remain {
                to_str(bgn).to_string() + to_str(s)
            } else {
                to_str(bgn).to_string()
            }
        ))
    ))
);

named!(bool_true<NodeKind>,
    ws!(do_parse!( tag!("true") >> (NodeKind::Bool(true)) ))
);

named!(bool_false<NodeKind>,
    ws!(do_parse!( tag!("false") >> (NodeKind::Bool(false)) ))
);

named!(constant<NodeKind>,
    alt_complete!(float | integer | ident | bool_false | bool_true)
);

named!(parens<NodeKind>, ws!(delimited!(tag!("("), expr, tag!(")"))));

named!(expr_prim<NodeKind>,
    alt!(constant | parens)
);

pub fn parse_simple_expr(e: &str) {
    println!("expr: {}\n{}", e, match expr(e.as_bytes()) {
        IResult::Done(_, expr_node) => format!("generated node: {:?}", expr_node),
        IResult::Incomplete(needed) => format!("imcomplete: {:?}",     needed),
        IResult::Error(err) =>         format!("error: {:?}",          err)
    });
}

#[test]
pub fn test_parse_simple_expr() {
    parse_simple_expr("5 / 2 + 11 * 10");
    parse_simple_expr("5.3 *. 10.2");
    parse_simple_expr(" a * (b + 3)");
}

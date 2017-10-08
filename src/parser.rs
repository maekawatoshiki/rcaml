use nom::{IResult, digit, double};

use std::str;
use std::str::FromStr;
use std::collections::HashMap;

use node;
use node::NodeKind;

use id::IdGen;

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

pub fn uniquify(expr: NodeKind, idgen: &mut IdGen) -> NodeKind {
    match expr {
        NodeKind::LetExpr((name, ty), expr, body) => {
            let ty = if let Type::Var(_) = ty {
                idgen.get_type()
            } else {
                ty
            };
            let expr = uniquify(*expr, idgen);
            let body = uniquify(*body, idgen);
            NodeKind::LetExpr((name, ty), Box::new(expr), Box::new(body))
        }
        NodeKind::LetDef((name, ty), expr) => {
            let ty = if let Type::Var(_) = ty {
                idgen.get_type()
            } else {
                ty
            };
            let expr = uniquify(*expr, idgen);
            NodeKind::LetDef((name, ty), Box::new(expr))
        }
        NodeKind::LetFuncExpr(node::FuncDef {
                                  name: (name, t),
                                  mut params,
                              },
                              expr,
                              body) => {
            let t = if let Type::Var(_) = t {
                idgen.get_type()
            } else {
                t
            };
            for &mut (_, ref mut param_ty) in &mut params {
                let entry = ::std::mem::replace(param_ty, Type::Unit);
                let new_ty = if let Type::Var(_) = entry {
                    idgen.get_type()
                } else {
                    entry
                };
                *param_ty = new_ty;
            }
            let expr = Box::new(uniquify(*expr, idgen));
            let body = Box::new(uniquify(*body, idgen));
            NodeKind::LetFuncExpr(
                node::FuncDef {
                    name: (name, t),
                    params: params,
                },
                expr,
                body,
            )
        }
        NodeKind::LetFuncDef(node::FuncDef {
                                 name: (name, t),
                                 mut params,
                             },
                             expr) => {
            let t = if let Type::Var(_) = t {
                idgen.get_type()
            } else {
                t
            };
            for &mut (_, ref mut param_ty) in &mut params {
                let entry = ::std::mem::replace(param_ty, Type::Unit);
                let new_ty = if let Type::Var(_) = entry {
                    idgen.get_type()
                } else {
                    entry
                };
                *param_ty = new_ty;
            }
            let expr = Box::new(uniquify(*expr, idgen));
            NodeKind::LetFuncDef(
                node::FuncDef {
                    name: (name, t),
                    params: params,
                },
                expr,
            )
        }
        NodeKind::IntBinaryOp(op, e1, e2) => {
            let e1 = Box::new(uniquify(*e1, idgen));
            let e2 = Box::new(uniquify(*e2, idgen));
            NodeKind::IntBinaryOp(op, e1, e2)
        }
        NodeKind::FloatBinaryOp(op, e1, e2) => {
            let e1 = Box::new(uniquify(*e1, idgen));
            let e2 = Box::new(uniquify(*e2, idgen));
            NodeKind::FloatBinaryOp(op, e1, e2)
        }
        NodeKind::Call(e1, mut e2s) => {
            let e1 = Box::new(uniquify(*e1, idgen));
            uniquify_seq(&mut e2s, idgen);
            NodeKind::Call(e1, e2s)
        }
        x => x, // No Syntax inside
    }
}

fn uniquify_seq(seq: &mut Vec<NodeKind>, id_gen: &mut IdGen) {
    for i in 0..seq.len() {
        let entry = ::std::mem::replace(&mut seq[i], NodeKind::Unit);
        seq[i] = uniquify(entry, id_gen);
    }
}

pub fn parse_and_show_simple_expr(e: &str) {
    println!("expr: {}\n{}", e, match expr(e.as_bytes()) {
        IResult::Done(_, expr_node) => format!("generated node: {:?}", expr_node),
        IResult::Incomplete(needed) => format!("imcomplete: {:?}",     needed),
        IResult::Error(err) =>         format!("error: {:?}",          err)
    });
}

pub fn parse_and_show_module_item(e: &str) {
    println!("module-item: {}\n{}", e, match module_item(e.as_bytes()) {
        IResult::Done(_, expr_node) => format!("generated node: {:?}", expr_node),
        IResult::Incomplete(needed) => format!("imcomplete: {:?}",     needed),
        IResult::Error(err) =>         format!("error: {:?}",          err)
    });
}

pub fn parse_module_items(e: &str) -> Vec<NodeKind> {
    use typing;
    use id;

    let mut idgen = id::IdGen::new();
    let mut tyenv = HashMap::new();
    let mut nodes = Vec::new();
    let mut code = e;
    while code.len() > 0 {
        match module_item(code.as_bytes()) {
            IResult::Done(remain, node) => {
                let uniquified = uniquify(node, &mut idgen);
                nodes.push(typing::f(&uniquified, &mut tyenv, &mut idgen));
                code = str::from_utf8(remain).unwrap();
            }
            IResult::Incomplete(needed) => panic!(format!("imcomplete: {:?}",     needed)),
            IResult::Error(err) => panic!(format!("error: {:?}",          err)),
        }
    }
    nodes
}

pub fn parse_and_infer_type(e: &str) {
    println!("expr: {}", e);
    let node = match module_item(e.as_bytes()) {
        IResult::Done(_, node) => node,
        _ => panic!(),
    };
    // sloppy impl of showing type-infered node
    use typing;
    use id;
    let mut idgen = id::IdGen::new();
    let mut tyenv = HashMap::new();
    let uniquified = uniquify(node, &mut idgen);
    println!("generated node: {:?}\ntype infered node: {:?}", uniquified, typing::f(&uniquified,&mut tyenv, &mut idgen));
}


use std::sync::Mutex;

lazy_static! {
    pub static ref EXTENV: Mutex<HashMap<String, Type>> = {
        let mut extenv = HashMap::new();
        extenv.insert("print_int".to_string(), 
                      Type::Func(vec![Type::Int], 
                      Box::new(Type::Unit)));
        Mutex::new(extenv)
    };
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

#[test]
pub fn test_parse_module_item() {
    use node::NodeKind::*;
    use node::FuncDef;
    use node::BinOps::*;
    use node::BinOps::*;

    let f = |e: &str| match module_item(e.as_bytes()) {
        IResult::Done(_, expr_node) => expr_node,
        IResult::Incomplete(needed) => panic!(format!("imcomplete: {:?}",     needed)),
        IResult::Error(err) => panic!(format!("error: {:?}", err)),
    };

    assert_eq!(f("let f x = x * 2;;"),
                LetFuncDef(
                    FuncDef { name: ("f".to_string(), Type::Var(0)), 
                              params: vec![("x".to_string(), Type::Var(0))] }, 
                              Box::new(IntBinaryOp(IMul, 
                                                   Box::new(Ident("x".to_string())), 
                                                   Box::new(Int(2))))));
    // assert_eq!(f("5.3 *. 10.2"),
    //            FloatBinaryOp(FMul,
    //                     Box::new(Float(5.3)),
    //                     Box::new(Float(10.2))))
}

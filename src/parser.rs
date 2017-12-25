use nom::{digit, double, IResult};

use std::str;
use std::str::FromStr;
use std::collections::HashMap;

extern crate rand;
use self::rand::Rng;

use node;
use node::NodeKind;
use closure::Prog;

use id::IdGen;

use typing::{Type, TypeScheme};

use std::boxed::Box;

// syntax reference: https://caml.inria.fr/pub/docs/manual-ocaml/language.html

pub fn to_str(slice: &[u8]) -> &str {
    str::from_utf8(slice).unwrap()
}

pub fn remove_comments(s: &[u8]) -> String {
    let mut level = 0;
    let mut pos = 0;
    let mut ret = "".to_string();
    let len = s.len();
    while pos < len {
        if pos < len - 1 && s[pos..(pos + 2)] == [b'(', b'*'] {
            pos += 2;
            level += 1;
            continue;
        }
        if pos < len - 1 && s[pos..(pos + 2)] == [b'*', b')'] {
            pos += 2;
            if level <= 0 {
                panic!("not found corresponding \"(*\"")
            }
            level -= 1;
            continue;
        }
        if level == 0 {
            ret.push(s[pos] as char);
        }
        pos += 1;
    }
    if level != 0 {
        panic!("comments are not balanced")
    }
    ret
}

named!(whitespace<()>, do_parse!(one_of!(" \t\n\r") >> ()));

named!(opt_spaces<()>, do_parse!(many0!(whitespace) >> ()));

named!(spaces<()>, do_parse!(many1!(whitespace) >> ()));

named!(
    funcdef<NodeKind>,
    do_parse!(
        name:   ident >> // TODO: not only identifier... (https://caml.inria.fr/pub/docs/manual-ocaml/patterns.html#pattern)
        params: many1!(do_parse!(spaces >> param: ident >> (param)))
            >> (NodeKind::FuncDef(
                (name.get_ident_name().unwrap(), Type::Var(0)),
                (params
                    .into_iter()
                    .map(|param| (param.get_ident_name().unwrap(), Type::Var(0)))
                    .collect())
            ))
    )
);

named!(
    pat<Vec<(String, Type)>>,
    do_parse!(
        init: ident_s
            >> res:
                fold_many1!(
                    ws!(preceded!(tag!(","), ident_s)),
                    vec![(init, Type::Var(0))],
                    |mut acc: Vec<(String, Type)>, x| {
                        acc.push((x, Type::Var(0)));
                        acc
                    }
                ) >> (res)
    )
);

named!(expr<NodeKind>, alt!(expr_let | expr_semicolon));

named!(
    expr_let<NodeKind>,
    alt_complete!(
        ws!(do_parse!(
            tag!("let") >>
        ws!(many0!(tag!("rec"))) >> // TODO: do not ignore rec
        name: alt!(funcdef | ident) >> ws!(tag!("=")) >> exp: expr >> tag!("in")
                >> body: expr >> (match name {
                NodeKind::FuncDef(name, params) => NodeKind::LetFuncExpr(
                    node::FuncDef {
                        name: name,
                        params: params,
                    },
                    Box::new(exp),
                    Box::new(body)
                ),
                NodeKind::Ident(name) => {
                    NodeKind::LetExpr((name, Type::Var(0)), Box::new(exp), Box::new(body))
                }
                _ => panic!(),
            })
        ))
            | ws!(do_parse!(
                tag!("let") >> tag!("(") >> p: pat >> tag!(")") >> ws!(tag!("=")) >> exp: expr
                    >> tag!("in") >> body: expr
                    >> (NodeKind::LetTupleExpr(p, Box::new(exp), Box::new(body)))
            ))
    )
);

named!(
    expr_semicolon<NodeKind>,
    ws!(do_parse!(
        init: expr_if
            >> res:
                fold_many0!(
                    do_parse!(op: tag!(";") >> rhs: expr >> (rhs)),
                    init,
                    |e1, e2: NodeKind| NodeKind::LetExpr(
                        ("_".to_string(), Type::Var(0)),
                        Box::new(e1),
                        Box::new(e2)
                    )
                ) >> (res)
    ))
);

named!(
    expr_if<NodeKind>,
    alt_complete!(
        ws!(do_parse!(
            tag!("if") >> e1: expr >> tag!("then") >> e2: expr >> tag!("else") >> e3: expr
                >> (NodeKind::IfExpr(Box::new(e1), Box::new(e2), Box::new(e3)))
        )) | expr_assign
    )
);

named!(
    expr_assign<NodeKind>,
    alt_complete!(
        ws!(do_parse!(
            base_indices: array_index_list >> tag!("<-") >> e: expr_comma >> ({
                let (mut base, mut indices) = base_indices;
                let last = indices.pop().unwrap(); // indices.len() >= 1
                for idx in indices {
                    base = NodeKind::Get(Box::new(base), Box::new(idx));
                }
                NodeKind::Put(Box::new(base), Box::new(last), Box::new(e))
            })
        )) | expr_comma
    )
);

named!(
    array_index_list<(NodeKind, Vec<NodeKind>)>,
    ws!(do_parse!(
        init: expr_prim
            >> res:
                fold_many1!(
                    ws!(do_parse!(
                        char!('.') >> char!('(') >> res: expr >> char!(')') >> (res)
                    )),
                    Vec::new(),
                    |mut acc: Vec<NodeKind>, index| {
                        acc.push(index);
                        acc
                    }
                ) >> ((init, res))
    ))
);

named!(
    expr_comma<NodeKind>,
    alt_complete!(
        ws!(do_parse!(
            init: expr_comp
                >> res:
                    fold_many1!(
                        do_parse!(tag!(",") >> rhs: expr_comp >> (rhs)),
                        vec![init],
                        |mut acc: Vec<NodeKind>, e| {
                            acc.push(e);
                            acc
                        }
                    ) >> (NodeKind::Tuple(res))
        )) | expr_comp
    )
);

named!(
    expr_comp<NodeKind>,
    ws!(do_parse!(
        init: expr_add_sub
            >> res:
                fold_many0!(
                    do_parse!(
                        op:
                            alt!(
                                tag!("<>") | tag!("==") | tag!("!=") | tag!("<=") | tag!(">=")
                                    | tag!("<") | tag!(">")
                                    | tag!("=")
                            ) >> rhs: expr_add_sub >> (op, rhs)
                    ),
                    init,
                    |n1, (op, n2): (&[u8], NodeKind)| NodeKind::CompBinaryOp(
                        node::str_to_comp_binop(str::from_utf8(op).unwrap()),
                        Box::new(n1),
                        Box::new(n2)
                    )
                ) >> (res)
    ))
);

named!(
    expr_add_sub<NodeKind>,
    ws!(do_parse!(
        init: expr_mul_div
            >> res:
                fold_many0!(
                    do_parse!(
                        op: alt!(tag!("+.") | tag!("-.") | tag!("+") | tag!("-"))
                            >> rhs: expr_mul_div >> (op, rhs)
                    ),
                    init,
                    |n1, (op, n2): (&[u8], NodeKind)| {
                        let (op, is_int) = node::str_to_binop(to_str(op));
                        if is_int {
                            NodeKind::IntBinaryOp(op, Box::new(n1), Box::new(n2))
                        } else {
                            NodeKind::FloatBinaryOp(op, Box::new(n1), Box::new(n2))
                        }
                    }
                ) >> (res)
    ))
);

named!(
    expr_mul_div<NodeKind>,
    ws!(do_parse!(
        init: expr_unary
            >> res:
                fold_many0!(
                    do_parse!(
                        op: alt!(tag!("mod") | tag!("*.") | tag!("/.") | tag!("*") | tag!("/"))
                            >> rhs: expr_unary >> (op, rhs)
                    ),
                    init,
                    |n1, (op, n2): (&[u8], NodeKind)| {
                        let (op, is_int) = node::str_to_binop(to_str(op));
                        if is_int {
                            NodeKind::IntBinaryOp(op, Box::new(n1), Box::new(n2))
                        } else {
                            NodeKind::FloatBinaryOp(op, Box::new(n1), Box::new(n2))
                        }
                    }
                ) >> (res)
    ))
);

named!(
    expr_unary<NodeKind>,
    ws!(alt!(
        do_parse!(
            op: alt!(tag!("-.") | tag!("-")) >> e: expr_unary >> ({
                let (op, is_int) = node::str_to_unaryop(str::from_utf8(op).unwrap());
                if is_int {
                    NodeKind::IntUnaryOp(op, Box::new(e))
                } else {
                    NodeKind::FloatUnaryOp(op, Box::new(e))
                }
            })
        ) | expr_postfix
    ))
);

named!(
    apply_postfix<Vec<NodeKind>>,
    do_parse!(spaces >> args: separated_nonempty_list_complete!(spaces, expr_prim) >> (args))
);

named!(
    expr_postfix<NodeKind>,
    alt_complete!(
        ws!(do_parse!(
            alt_complete!(tag!("Array.create") | tag!("Array.make")) >> res: many1!(ws!(expr_prim))
                >> ({
                    let mut res = res;
                    if res.len() != 2 {
                        panic!("The number of Array.create is wrong");
                    }
                    let res1 = res.pop().unwrap();
                    let res0 = res.pop().unwrap();
                    NodeKind::MakeArray(Box::new(res0), Box::new(res1))
                })
        ))
            | ws!(do_parse!(
                init: expr_prim >> folded: fold_many0!(apply_postfix, init, |lhs, pf| {
                    NodeKind::Call(Box::new(lhs), pf)
                }) >> (folded)
            ))
    )
);

named!(expr_prim<NodeKind>, alt!(constant | parens | unit));

named!(
    integer<NodeKind>,
    do_parse!(
        i: map_res!(map_res!(digit, str::from_utf8), FromStr::from_str) >> (NodeKind::Int(i))
    )
);

named!(
    float<NodeKind>,
    do_parse!(f: double >> (NodeKind::Float(f)))
);

fn is_ident(x: &[u8]) -> bool {
    let keywords = vec![
        &b"let"[..],
        &b"rec"[..],
        &b"in"[..],
        &b"true"[..],
        &b"false"[..],
        &b"if"[..],
        &b"then"[..],
        &b"else"[..],
        &b"Array.create"[..],
        &b"Array.make"[..],
        &b"mod"[..],
    ];
    if x.len() == 0 || keywords.contains(&x) {
        return false;
    }
    !(b'0' <= x[0] && x[0] <= b'9')
}
fn is_not_ident_u8(x: u8) -> bool {
    !((b'0' <= x && x <= b'9') || (b'A' <= x && x <= b'Z') || (b'a' <= x && x <= b'z') || x == b'_')
}

named!(
    ident_s<String>,
    do_parse!(
        i: verify!(take_till!(is_not_ident_u8), is_ident)
            >> (String::from_utf8(i.to_vec()).unwrap())
    )
);
named!(
    ident<NodeKind>,
    do_parse!(
        i: verify!(take_till!(is_not_ident_u8), is_ident)
            >> (NodeKind::Ident(String::from_utf8(i.to_vec()).unwrap()))
    )
);

named!(
    bool_true<NodeKind>,
    do_parse!(tag!("true") >> (NodeKind::Bool(true)))
);

named!(
    bool_false<NodeKind>,
    do_parse!(tag!("false") >> (NodeKind::Bool(false)))
);

named!(
    constant<NodeKind>,
    alt_complete!(float | integer | ident | bool_false | bool_true)
);

named!(
    parens<NodeKind>,
    delimited!(tag!("("), ws!(expr), tag!(")"))
);

named!(
    unit<NodeKind>,
    do_parse!(tag!("(") >> tag!(")") >> (NodeKind::Unit))
);

named!(opt_dscolon<()>, do_parse!(many0!(tag!(";;")) >> ()));

#[macro_export]
named!(pub module_item<NodeKind>,
    ws!(do_parse!(
        i: alt!(expr | definition) >> 
        opt_dscolon >> (i)
    ))
);

named!(definition<NodeKind>, alt!(ws!(definition_let)));

named!(
    definition_let<NodeKind>,
    ws!(do_parse!(
        tag!("let") >> name: alt!(funcdef | ident) >> ws!(tag!("=")) >> exp: expr >> (match name {
            NodeKind::FuncDef(name, params) => NodeKind::LetFuncDef(
                node::FuncDef {
                    name: name,
                    params: params,
                },
                Box::new(exp)
            ),
            NodeKind::Ident(name) => NodeKind::LetDef((name, Type::Var(0)), Box::new(exp)),
            _ => panic!(),
        })
    ))
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
        NodeKind::LetFuncExpr(
            node::FuncDef {
                name: (name, t),
                mut params,
            },
            expr,
            body,
        ) => {
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
        NodeKind::LetTupleExpr(mut pat, expr, body) => {
            for i in 0..pat.len() {
                if let Type::Var(_) = pat[i].1 {
                    pat[i].1 = idgen.get_type();
                }
            }
            NodeKind::LetTupleExpr(
                pat,
                Box::new(uniquify(*expr, idgen)),
                Box::new(uniquify(*body, idgen)),
            )
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
        NodeKind::LetFuncDef(
            node::FuncDef {
                name: (name, t),
                mut params,
            },
            expr,
        ) => {
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
            if let &NodeKind::Ident(_) = &*e1 {
                let e1 = Box::new(uniquify(*e1, idgen));
                uniquify_seq(&mut e2s, idgen);
                NodeKind::Call(e1, e2s)
            } else {
                let rand_name: String = rand::thread_rng().gen_ascii_chars().take(8).collect();
                uniquify_seq(&mut e2s, idgen);
                NodeKind::LetExpr(
                    (rand_name.clone(), idgen.get_type()),
                    e1,
                    Box::new(NodeKind::Call(Box::new(NodeKind::Ident(rand_name)), e2s)),
                )
            }
        }
        NodeKind::IfExpr(c, t, e) => NodeKind::IfExpr(
            Box::new(uniquify(*c, idgen)),
            Box::new(uniquify(*t, idgen)),
            Box::new(uniquify(*e, idgen)),
        ),
        NodeKind::MakeArray(e1, e2) => {
            let e1 = Box::new(uniquify(*e1, idgen));
            let e2 = Box::new(uniquify(*e2, idgen));
            NodeKind::MakeArray(e1, e2)
        }
        NodeKind::Get(e1, e2) => {
            let e1 = Box::new(uniquify(*e1, idgen));
            let e2 = Box::new(uniquify(*e2, idgen));
            NodeKind::Get(e1, e2)
        }
        NodeKind::Put(e1, e2, e3) => {
            let e1 = Box::new(uniquify(*e1, idgen));
            let e2 = Box::new(uniquify(*e2, idgen));
            let e3 = Box::new(uniquify(*e3, idgen));
            NodeKind::Put(e1, e2, e3)
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
    println!(
        "expr: {}\n{}",
        e,
        match expr(e.as_bytes()) {
            IResult::Done(_, expr_node) => format!("generated node: {:?}", expr_node),
            IResult::Incomplete(needed) => format!("imcomplete: {:?}", needed),
            IResult::Error(err) => format!("error: {:?}", err),
        }
    );
}

pub fn parse_and_show_module_item(e: &str) {
    println!(
        "module-item: {}\n{}",
        e,
        match module_item(e.as_bytes()) {
            IResult::Done(_, expr_node) => format!("generated node: {:?}", expr_node),
            IResult::Incomplete(needed) => format!("imcomplete: {:?}", needed),
            IResult::Error(err) => format!("error: {:?}", err),
        }
    );
}

extern crate ansi_term;
use self::ansi_term::{Colour, Style};

// TODO: this func parses code and even runs program
pub fn parse_module_items(e: &str) -> Vec<Prog> {
    use typing;
    use id;
    use codegen;
    use closure;

    let mut idgen = id::IdGen::new();
    let mut tyenv = HashMap::new();
    let mut progs = Vec::new();
    let e = remove_comments(e.as_bytes());
    let mut code = e.as_str();

    println!(
        "{}",
        Style::new()
            .underline()
            .bold()
            .paint(format!("expression:\t{}", code))
    );

    while code.len() > 0 {
        match module_item(code.as_bytes()) {
            IResult::Done(remain, node) => {
                let uniquified = uniquify(node, &mut idgen);
                println!("{:?}", uniquified.clone());
                let infered = typing::f(&uniquified, &mut tyenv, &mut idgen);
                let closured = closure::f(infered);
                println!(
                    "{}",
                    Colour::Green
                        .bold()
                        .paint(format!("program:\t{:?}", closured))
                );
                progs.push(closured);
                code = str::from_utf8(remain).unwrap();
            }
            IResult::Incomplete(needed) => panic!(format!("imcomplete: {:?}", needed)),
            IResult::Error(err) => panic!(format!("error: {:?}", err)),
        }
    }

    unsafe {
        let mut codegen = codegen::CodeGen::new(&mut tyenv);
        codegen.gen(true, true, progs.clone()).unwrap();
    }
    progs
}

pub fn do_parse_typing_closure(e: &str) -> (Vec<Prog>, HashMap<usize, Type>) {
    use typing;
    use id;
    use closure;

    let mut idgen = id::IdGen::new();
    let mut tyenv = HashMap::new();
    let mut progs = Vec::new();
    let e = remove_comments(e.as_bytes());
    let mut code = e.as_str();

    while code.len() > 0 {
        match module_item(code.as_bytes()) {
            IResult::Done(remain, node) => {
                let uniquified = uniquify(node, &mut idgen);
                let infered = typing::f(&uniquified, &mut tyenv, &mut idgen);
                let closured = closure::f(infered);
                progs.push(closured);
                code = str::from_utf8(remain).unwrap();
            }
            IResult::Incomplete(needed) => panic!(format!("imcomplete: {:?}", needed)),
            IResult::Error(err) => panic!(format!("error: {:?}", err)),
        }
    }

    (progs, tyenv)
}

pub fn parse_and_infer_type(e: &str) {
    println!(
        "{}",
        Style::new()
            .underline()
            .bold()
            .paint(format!("expression:\t{}", e))
    );
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
    println!(
        "{}",
        Colour::Yellow
            .bold()
            .paint(format!("generated:\t{:?}", uniquified))
    );
    println!(
        "{}",
        Colour::Green.bold().paint(format!(
            "infered:\t{:?}",
            typing::f(&uniquified, &mut tyenv, &mut idgen)
        ))
    );
}

pub fn parse_and_infer_type_and_closure_conv(e: &str) {
    println!(
        "{}",
        Style::new()
            .underline()
            .bold()
            .paint(format!("expression:\t{}", e))
    );
    let node = match module_item(e.as_bytes()) {
        IResult::Done(_, node) => node,
        _ => panic!(),
    };
    // sloppy impl of showing type-infered node
    use typing;
    use id;
    use closure;
    let mut idgen = id::IdGen::new();
    let mut tyenv = HashMap::new();
    let uniquified = uniquify(node, &mut idgen);
    let infered = typing::f(&uniquified, &mut tyenv, &mut idgen);
    let closured = closure::f(infered);
    println!(
        "{}",
        Colour::Yellow
            .bold()
            .paint(format!("generated:\t{:?}", uniquified))
    );
    println!(
        "{}",
        Colour::Green
            .bold()
            .paint(format!("closure:\t{:?}", closured))
    );
}

use std::sync::Mutex;

lazy_static! {
    pub static ref EXTENV: Mutex<HashMap<String, TypeScheme>> = {
        let mut extenv = HashMap::new();
        extenv.insert("print_int".to_string(),
                      TypeScheme::new(vec![], Type::Func(
                                                vec![Type::Int],
                                                Box::new(Type::Unit))));
        extenv.insert("print_float".to_string(),
                      TypeScheme::new(vec![], Type::Func(
                                                vec![Type::Float],
                                                Box::new(Type::Unit))));
        extenv.insert("print_newline".to_string(),
                      TypeScheme::new(vec![], Type::Func(
                                                vec![Type::Unit],
                                                Box::new(Type::Unit))));
        extenv.insert("float_of_int".to_string(),
                      TypeScheme::new(vec![], Type::Func(
                                                vec![Type::Int],
                                                Box::new(Type::Float))));
        Mutex::new(extenv)
    };
}

#[test]
pub fn test_parse_simple_expr() {
    use node::NodeKind::*;
    use node::BinOps::*;

    let f = |e: &str| match expr(e.as_bytes()) {
        IResult::Done(_, expr_node) => expr_node,
        IResult::Incomplete(needed) => panic!(format!("imcomplete: {:?}", needed)),
        IResult::Error(err) => panic!(format!("error: {:?}", err)),
    };

    assert_eq!(
        f("5 / a3 + 11 * 10"),
        IntBinaryOp(
            IAdd,
            Box::new(IntBinaryOp(
                IDiv,
                Box::new(Int(5)),
                Box::new(Ident("a3".to_string()))
            )),
            Box::new(IntBinaryOp(IMul, Box::new(Int(11)), Box::new(Int(10))))
        )
    );
    assert_eq!(
        f("5.3 *. 10.2"),
        FloatBinaryOp(FMul, Box::new(Float(5.3)), Box::new(Float(10.2)))
    )
}

#[test]
pub fn test_parse_module_item() {
    use node::NodeKind::*;
    use node::FuncDef;
    use node::BinOps::*;

    let f = |e: &str| match module_item(e.as_bytes()) {
        IResult::Done(_, expr_node) => expr_node,
        IResult::Incomplete(needed) => panic!(format!("imcomplete: {:?}", needed)),
        IResult::Error(err) => panic!(format!("error: {:?}", err)),
    };

    assert_eq!(
        f("let f x = x * 2;;"),
        LetFuncDef(
            FuncDef {
                name: ("f".to_string(), Type::Var(0)),
                params: vec![("x".to_string(), Type::Var(0))],
            },
            Box::new(IntBinaryOp(
                IMul,
                Box::new(Ident("x".to_string())),
                Box::new(Int(2))
            ))
        )
    );
}

#[test]
pub fn test_type_infer() {
    use node::NodeKind::*;
    use node::FuncDef;
    use node::BinOps::*;
    use node::BinOps::*;
    use id::IdGen;
    use typing;

    let mut idgen = IdGen::new();

    let mut f = |e: &str| match module_item(e.as_bytes()) {
        IResult::Done(remain, node) => {
            let uniquified = uniquify(node, &mut idgen);
            typing::g(
                &uniquified,
                &HashMap::new(),
                &mut HashMap::new(),
                &mut idgen,
            )
        }
        IResult::Incomplete(needed) => panic!(format!("imcomplete: {:?}", needed)),
        IResult::Error(err) => panic!(format!("error: {:?}", err)),
    };

    assert_eq!(
        f("let id x = x in let f y = id (y id) in let f = f in f")
            .unwrap()
            .to_string(),
        "((('1 -> '1) -> '2) -> '2)"
    );
}

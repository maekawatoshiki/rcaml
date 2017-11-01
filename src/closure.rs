use node::NodeKind;
use node;
use typing::Type;
use node::{BinOps, CompBinOps};

use std::collections::{HashMap, hash_map, HashSet};

extern crate ordered_float;
use self::ordered_float::OrderedFloat;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cls {
    pub entry: String,
    pub actual_fv: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Closure {
    Unit,
    Bool(bool),
    Int(i32),
    Float(OrderedFloat<f64>),
    Var(String),
    IntBinaryOp(BinOps, Box<Closure>, Box<Closure>),
    FloatBinaryOp(BinOps, Box<Closure>, Box<Closure>),
    CompBinaryOp(CompBinOps, Box<Closure>, Box<Closure>),
    AppCls(Box<Closure>, Vec<Closure>),
    AppDir(Box<Closure>, Vec<Closure>),
    LetExpr((String, Type), Box<Closure>, Box<Closure>), // (name, ty), bound expr, body
    If(Box<Closure>, Box<Closure>, Box<Closure>), // cond, then, else
    MakeCls(String, Type, Cls, Box<Closure>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncDef {
    pub name: (String, Type),
    pub params: Vec<(String, Type)>,
    pub formal_fv: Vec<(String, Type)>,
    pub body: Box<Closure>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prog(pub Vec<FuncDef>, pub Closure);

macro_rules! build_set {
    ($($x:expr),*) => ({
        let mut h = HashSet::new();
        $(h.insert($x.clone());)*
            h
    })
}
fn fv(e: &Closure) -> HashSet<String> {
    use self::Closure::*;
    macro_rules! seq {
        ($e:expr) => { {
            let mut v = Vec::new();
            for y in $e {
                for e in fv(&y) {
                    v.push(e);
                }
            }
            v
        } }
    }
    match *e {
        Unit | Int(_) | Float(_) => HashSet::new(),
        // Neg(ref x) | FNeg(ref x) => build_set!(x),
        IntBinaryOp(_, ref x, ref y) |
        FloatBinaryOp(_, ref x, ref y) | 
        CompBinaryOp(_, ref x , ref y) => {
            let mut set = HashSet::new();
            for e in fv(x).union(&fv(y)).collect::<Vec<&String>>() {
                set.insert(e.clone());
            }
            set
        }
        // Get(ref x, ref y) => build_set!(x, y),
        If(ref c, ref t, ref e) => {
            let c = fv(c);
            let t = fv(t);
            let e = fv(e);
            &(&c | &t) | &e
        }
        LetExpr((ref x, _), ref e1, ref e2) => {
            let s1 = fv(e1);
            let s2 = &fv(e2) - &build_set!(x);
            &s1 | &s2
        }
        Var(ref x) => build_set!(x),
        MakeCls(ref x,
                _,
                Cls {
                    entry: _,
                    actual_fv: ref ys,
                },
                ref e) => &(&ys.iter().cloned().collect() | &fv(e)) - &build_set!(x),
        AppCls(ref x, ref args) => {
            &fv(x) | &seq!(args).iter().map(|v| (*v).clone()).collect::<HashSet<_>>()
        }
        AppDir(_, ref xs)
        // Tuple(ref xs)
        => {
            seq!(xs).iter().map(|y| (*y).clone()).collect::<HashSet<_>>()
        }
        // LetTuple(ref xs, ref y, ref e)
        // => {
        //     let tmp: HashSet<String> = xs.iter().map(|x| x.0.clone()).collect(); // S.of_list (List.map fst xs)
        //     &build_set!(y) | &(&invoke!(e) - &tmp)
        // }
        // Put(ref x, ref y, ref z) => build_set!(x, y, z),
        _ => panic!(format!("{:?}", e)),
    }
}

fn g(
    node: NodeKind,
    env: &HashMap<String, Type>,
    known: &HashSet<String>,
    toplevel: &mut Vec<FuncDef>,
) -> Closure {
    macro_rules! seq { ($e:expr) => { {
        let mut a = Vec::new();
        for c in $e {
          a.push(  g(c, env, known , toplevel))
        }
        a }
    }};
    match node {
        NodeKind::Unit => Closure::Unit,
        NodeKind::Int(i) => Closure::Int(i),
        NodeKind::Float(f) => Closure::Float(OrderedFloat::from(f)),
        NodeKind::Ident(name) => Closure::Var(name),
        NodeKind::IntBinaryOp(op, lhs, rhs) => {
            Closure::IntBinaryOp(
                op,
                Box::new(g(*lhs, env, known, toplevel)),
                Box::new(g(*rhs, env, known, toplevel)),
            )
        }
        NodeKind::FloatBinaryOp(op, lhs, rhs) => {
            Closure::FloatBinaryOp(
                op,
                Box::new(g(*lhs, env, known, toplevel)),
                Box::new(g(*rhs, env, known, toplevel)),
            )
        }
        NodeKind::CompBinaryOp(op, lhs, rhs) => {
            Closure::CompBinaryOp(
                op,
                Box::new(g(*lhs, env, known, toplevel)),
                Box::new(g(*rhs, env, known, toplevel)),
            )
        }
        NodeKind::IfExpr(cond, then, els) => {
            Closure::If(
                Box::new(g(*cond, env, known, toplevel)),
                Box::new(g(*then, env, known, toplevel)),
                Box::new(g(*els, env, known, toplevel)),
            )
        }
        // LetExpr((String, typing::Type), Box<NodeKind>, Box<NodeKind>), // (name, ty), bound expr, body
        NodeKind::LetExpr((name, ty), expr, body) => {
            let mut cp_env = env.clone();
            cp_env.insert(name.clone(), ty.clone());
            Closure::LetExpr(
                (name, ty),
                Box::new(g(*expr, env, known, toplevel)),
                Box::new(g(*body, &cp_env, known, toplevel)),
            )
        }

        NodeKind::LetFuncExpr(node::FuncDef {
                                  name: (x, t),
                                  params: params,
                              },
                              expr,
                              body) => {
            // /* Follow the original code */
            let mut toplevel_cp = toplevel.clone();
            let mut env_p = env.clone();
            env_p.insert(x.clone(), t.clone());
            let mut known_p = known.clone();
            known_p.insert(x.clone());
            let mut env_p2 = env_p.clone();
            for &(ref y, ref t) in params.iter() {
                env_p2.insert(y.clone(), t.clone());
            }
            let e1p = g((*expr).clone(), &env_p2, &known_p, &mut toplevel_cp);
            /* Check if e1p contains free variables */
            let zs = &fv(&e1p) - &params.iter().map(|&(ref y, _)| y.clone()).collect();
            let (known_p, e1p) = if zs.is_empty() {
                *toplevel = toplevel_cp;
                (&known_p, e1p)
            } else {
                let e1p = g(*expr, &env_p2, known, toplevel);
                (known, e1p)
            };
            let zs: Vec<String> = (&zs - &build_set!(x)).into_iter().collect();
            let zts: Vec<(String, Type)> = zs.iter()
                .map(|&ref z| (z.clone(), env.get(z).unwrap().clone()))
                .collect();
            toplevel.push(FuncDef {
                name: (x.clone(), t.clone()),
                params: params,
                formal_fv: zts,
                body: Box::new(e1p),
            });
            let e2p = g(*body, &env_p, known_p, toplevel);
            if fv(&e2p).contains(&x) {
                Closure::MakeCls(
                    x.clone(),
                    t,
                    Cls {
                        entry: x,
                        actual_fv: zs,
                    },
                    Box::new(e2p),
                )
            } else {
                e2p
            }
        }

        NodeKind::Call(callee, args) => {
            let name = if let NodeKind::Ident(name) = *callee.clone() {
                name
            } else {
                panic!()
            };
            if known.contains(&name) {
                Closure::AppDir(Box::new(g(*callee, env, known, toplevel)), seq!(args))
            } else {
                Closure::AppCls(Box::new(g(*callee, env, known, toplevel)), seq!(args))
            }
        }

        _ => panic!(),
    }
}

pub fn f(e: NodeKind) -> Prog {
    let mut toplevel = Vec::new();
    let e = g(e, &HashMap::new(), &HashSet::new(), &mut toplevel);
    Prog(toplevel, e)
}

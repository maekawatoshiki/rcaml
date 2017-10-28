use node::*;
use typing::Type;
use node::BinOps;

use std::collections::{HashMap, hash_map, HashSet};

pub enum Closure {
    Unit,
    Bool(bool),
    Int(i32),
    Float(f64),
    Var(String),
    IntBinaryOp(BinOps, Box<Closure>, Box<Closure>),
    FloatBinaryOp(BinOps, Box<Closure>, Box<Closure>),
    AppCls(Box<Closure>, Vec<Closure>),
    AppDir(Box<Closure>, Vec<Closure>),
}

fn g(node: NodeKind, env: &HashMap<String, Type>, known: &HashSet<String>) -> Closure {
    match node {
        NodeKind::Unit => Closure::Unit,
        NodeKind::Int(i) => Closure::Int(i),
        NodeKind::Float(f) => Closure::Float(f),
        NodeKind::Ident(name) => Closure::Var(name),
        NodeKind::IntBinaryOp(op, lhs, rhs) => {
            Closure::IntBinaryOp(
                op,
                Box::new(g(*lhs, env, known)),
                Box::new(g(*rhs, env, known)),
            )
        }
        NodeKind::FloatBinaryOp(op, lhs, rhs) => {
            Closure::FloatBinaryOp(
                op,
                Box::new(g(*lhs, env, known)),
                Box::new(g(*rhs, env, known)),
            )
        }
        _ => panic!(),
    }
}

pub fn f(e: NodeKind) -> Closure {
    g(e, &HashMap::new(), &HashSet::new())
}

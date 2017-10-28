use node::*;
use typing::Type;

use std::collections::{HashMap, hash_map, HashSet};

pub enum Closure {
    Unit,
    Bool(bool),
    Int(i32),
    Float(f64),
    Var(String),
    AppCls(Box<Closure>, Vec<Closure>),
    AppDir(Box<Closure>, Vec<Closure>),
}

fn g(node: NodeKind, env: &HashMap<String, Type>, known: &HashSet<String>) -> Closure {
    match node {
        NodeKind::Unit => Closure::Unit,
        NodeKind::Int(i) => Closure::Int(i),
        NodeKind::Float(f) => Closure::Float(f),
        _ => panic!(),
    }
}

pub fn f(e: NodeKind) -> Closure {
    g(e, &HashMap::new(), &HashSet::new())
}

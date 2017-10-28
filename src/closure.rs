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

fn g(env: &HashMap<String, Type>, known: &HashSet<String>, e: NodeKind) -> Closure {
    Closure::Unit
}

pub fn f(e: NodeKind) -> Closure {
    g(&HashMap::new(), &HashSet::new(), e)
}

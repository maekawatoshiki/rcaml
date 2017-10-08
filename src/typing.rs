use std::boxed::Box;
use std::collections::HashMap;

use node::{NodeKind, FuncDef};
use id;

use parser::EXTENV;

#[derive(Clone, PartialEq)]
pub enum Type {
    Unit,
    Bool,
    Int,
    Float,
    Char,
    Func(Vec<Type>, Box<Type>), // (param types, return type)
    Var(usize), // id
}

impl Type {
    pub fn to_string(&self) -> String {
        match self {
            &Type::Unit => "unit".to_string(),
            &Type::Bool => "bool".to_string(),
            &Type::Char => "char".to_string(),
            &Type::Int => "int".to_string(),
            &Type::Float => "float".to_string(),
            &Type::Func(ref param_tys, ref ret_ty) => {
                param_tys.into_iter().fold("".to_string(), |acc, ts| {
                    acc + ts.to_string().as_str() + " -> "
                }) + ret_ty.to_string().as_str() + " = <fun>"
            }
            &Type::Var(id) => format!("var({})", id),
        }
    }
}

use std::fmt;
impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

#[derive(Debug)]
pub enum TypeError {
    Unify(Type, Type),
}

fn deref_ty(ty: &Type, tyenv: &HashMap<usize, Type>) -> Type {
    macro_rules! deref_typ_list {
        ($list:expr) => ($list.iter().map(|x| deref_ty(x, tyenv))
                                  .collect::<Vec<_>>());
    }
    match *ty {
        Type::Func(ref p, ref r) => Type::Func(deref_typ_list!(p), Box::new(deref_ty(r, tyenv))),
        // Type::Tuple(ref ts) => Type::Tuple(deref_ty_list!(ts)),
        // Type::Array(ref t) => Type::Array(Box::new(deref_ty(t, tyenv))),
        Type::Var(ref n) => {
            if let Some(t) = tyenv.get(n) {
                deref_ty(t, tyenv)
            } else {
                Type::Var(*n)
                // panic!("uninstantiated variable is not allowed now")
            }
        }
        _ => ty.clone(),
    }
}

fn deref_term(node: &NodeKind, tyenv: &HashMap<usize, Type>) -> NodeKind {
    macro_rules! apply_deref {
        ($ary:expr) => ($ary.iter().map(|x| deref_term(x, tyenv)).collect::<Vec<_>>());
    }
    match *node {
        NodeKind::IntBinaryOp(ref op, ref lhs, ref rhs) => {
            NodeKind::IntBinaryOp(
                op.clone(),
                Box::new(deref_term(&**lhs, tyenv)),
                Box::new(deref_term(&**rhs, tyenv)),
            )
        }
        NodeKind::FloatBinaryOp(ref op, ref lhs, ref rhs) => {
            NodeKind::FloatBinaryOp(
                op.clone(),
                Box::new(deref_term(&**lhs, tyenv)),
                Box::new(deref_term(&**rhs, tyenv)),
            )
        }
        NodeKind::Call(ref e, ref args) => {
            NodeKind::Call(Box::new(deref_term(e, tyenv)), apply_deref!(args))
        }
        NodeKind::LetExpr((ref name, ref ty), ref expr, ref body) => {
            NodeKind::LetExpr(
                (name.clone(), deref_ty(ty, tyenv)),
                Box::new(deref_term(&**expr, tyenv)),
                Box::new(deref_term(&**body, tyenv)),
            )
        }
        NodeKind::LetDef((ref name, ref ty), ref expr) => {
            NodeKind::LetDef(
                (name.clone(), deref_ty(ty, tyenv)),
                Box::new(deref_term(&**expr, tyenv)),
            )
        }
        NodeKind::LetFuncExpr(ref funcdef, ref expr, ref body) => {
            let (ref name, ref ty) = funcdef.name;
            let params = &funcdef.params;
            NodeKind::LetFuncExpr(
                FuncDef {
                    name: (name.to_string(), deref_ty(ty, tyenv)),
                    params: params
                        .iter()
                        .map(|&(ref x, ref t)| (x.clone(), deref_ty(t, tyenv)))
                        .collect::<Vec<_>>(),
                },
                Box::new(deref_term(expr, tyenv)),
                Box::new(deref_term(body, tyenv)),
            )
        }
        NodeKind::LetFuncDef(ref funcdef, ref expr) => {
            let (ref name, ref ty) = funcdef.name;
            let params = &funcdef.params;
            NodeKind::LetFuncDef(
                FuncDef {
                    name: (name.to_string(), deref_ty(ty, tyenv)),
                    params: params
                        .iter()
                        .map(|&(ref x, ref t)| (x.clone(), deref_ty(t, tyenv)))
                        .collect::<Vec<_>>(),
                },
                Box::new(deref_term(expr, tyenv)),
            )
        }
        _ => node.clone(),
    }
}

fn occur(r1: usize, ty: &Type) -> bool {
    macro_rules! occur_list {
        ($ls:expr) => ($ls.iter().any(|ty| occur(r1, ty)))
    }
    match *ty {
        Type::Func(ref t2s, ref t2) => occur_list!(t2s) || occur(r1, t2),
        // Type::Tuple(ref t2s) => occur_list!(t2s),
        // Type::Array(ref t2) => occur(r1, t2),
        Type::Var(r2) => r1 == r2,
        _ => false,
    }
}

pub fn unify(t1: &Type, t2: &Type, tyenv: &mut HashMap<usize, Type>) -> Result<(), TypeError> {
    match (t1, t2) {
        (&Type::Bool, &Type::Bool) => Ok(()),
        (&Type::Char, &Type::Char) => Ok(()),
        (&Type::Int, &Type::Int) => Ok(()),
        (&Type::Float, &Type::Float) => Ok(()),
        (&Type::Func(ref t1p, ref t1r), &Type::Func(ref t2p, ref t2r)) => {
            if t1p.len() != t2p.len() {
                return Err(TypeError::Unify(t1.clone(), t2.clone()));
            }
            for (a, b) in t1p.iter().zip(t2p.iter()) {
                try!(unify(a, b, tyenv));
            }
            unify(t1r, t2r, tyenv)
        }
        (&Type::Var(i1), &Type::Var(i2)) if i1 == i2 => Ok(()),
        (&Type::Var(ref i1), _) => {
            if let Some(t1sub) = tyenv.get(i1).cloned() {
                unify(&t1sub, t2, tyenv)
            } else {
                if occur(*i1, t2) {
                    return Err(TypeError::Unify(t1.clone(), t2.clone()));
                }
                tyenv.insert(*i1, t2.clone());
                Ok(())
            }
        }
        (_, &Type::Var(_)) => unify(t2, t1, tyenv),
        // TODO: implement more types
        _ => Err(TypeError::Unify(t1.clone(), t2.clone())),
    }
}

pub fn g(
    node: &NodeKind,
    env: &HashMap<String, Type>,
    tyenv: &mut HashMap<usize, Type>,
    idgen: &mut id::IdGen,
) -> Result<Type, TypeError> {
    macro_rules! g_seq {
        ($es:expr) => ({
            let mut argtys = Vec::new();
            for e in $es.iter() { argtys.push(try!(g(e, env, tyenv, idgen))); }
            argtys 
        });
    }
    match *node {
        NodeKind::Bool(_) => Ok(Type::Bool),
        NodeKind::Int(_) => Ok(Type::Int),
        NodeKind::Float(_) => Ok(Type::Float),
        NodeKind::Ident(ref name) => {
            if let Some(t) = env.get(name).cloned() {
                Ok(t)
            } else if let Some(t) = EXTENV.lock().unwrap().get(name).cloned() {
                Ok(t)
            } else {
                panic!("TODO: implement")
            }
        }
        NodeKind::IntBinaryOp(_, ref lhs, ref rhs) => {
            try!(unify(&try!(g(lhs, env, tyenv, idgen)), &Type::Int, tyenv));
            try!(unify(&try!(g(rhs, env, tyenv, idgen)), &Type::Int, tyenv));
            Ok(Type::Int)
        }
        NodeKind::FloatBinaryOp(_, ref lhs, ref rhs) => {
            try!(unify(&try!(g(lhs, env, tyenv, idgen)), &Type::Float, tyenv));
            try!(unify(&try!(g(rhs, env, tyenv, idgen)), &Type::Float, tyenv));
            Ok(Type::Float)
        }
        NodeKind::Call(ref callee, ref args) => {
            let ty = idgen.get_type();
            let functy = Type::Func(g_seq!(args), Box::new(ty.clone()));
            try!(unify(&try!(g(callee, env, tyenv, idgen)), &functy, tyenv));
            Ok(ty)
        }
        NodeKind::LetExpr((ref name, ref ty), ref expr, ref body) => {
            try!(unify(&try!(g(expr, env, tyenv, idgen)), ty, tyenv));
            let mut newenv = HashMap::new();
            newenv.insert(name.clone(), ty.clone());
            g(body, &newenv, tyenv, idgen)
        }
        NodeKind::LetFuncExpr(ref funcdef, ref expr, ref body) => {
            let (name, ty) = funcdef.name.clone();
            let params = &funcdef.params;
            let mut newenv = env.clone();
            newenv.insert(name.clone(), ty.clone());
            let mut newenv_body = newenv.clone();
            for &(ref x, ref t) in params.iter() {
                newenv_body.insert(x.to_string(), t.clone());
            }
            try!(unify(
                &ty,
                &Type::Func(
                    params.iter().map(|p| p.1.clone()).collect::<Vec<_>>(),
                    Box::new(try!(g(expr, &newenv_body, tyenv, idgen))),
                ),
                tyenv,
            ));
            g(body, &newenv, tyenv, idgen)
        }
        NodeKind::LetDef((ref name, ref ty), ref expr) => {
            try!(unify(&try!(g(expr, env, tyenv, idgen)), ty, tyenv));
            Ok(Type::Unit)
        }
        NodeKind::LetFuncDef(ref funcdef, ref expr) => {
            let (name, ty) = funcdef.name.clone();
            let params = &funcdef.params;
            let mut newenv = env.clone();
            newenv.insert(name.clone(), ty.clone());
            let mut newenv_body = newenv.clone();
            for &(ref x, ref t) in params.iter() {
                newenv_body.insert(x.to_string(), t.clone());
            }
            try!(unify(
                &ty,
                &Type::Func(
                    params.iter().map(|p| p.1.clone()).collect::<Vec<_>>(),
                    Box::new(try!(g(expr, &newenv_body, tyenv, idgen))),
                ),
                tyenv,
            ));
            EXTENV.lock().unwrap().insert(name.clone(), ty.clone());
            Ok(Type::Unit)
        }
        _ => panic!(),
    }
}

pub fn f(node: &NodeKind, tyenv: &mut HashMap<usize, Type>, idgen: &mut id::IdGen) -> NodeKind {
    let infered_ty = g(node, &HashMap::new(), tyenv, idgen);
    // TODO: originally maybe infered_ty must be Type::Unit
    deref_term(node, &tyenv)
}

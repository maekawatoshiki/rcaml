use node::NodeKind;
use node;
use typing::Type;
use node::{BinOps, CompBinOps};
use closure::{Closure, Prog, FuncDef};

use std::collections::{HashMap, HashSet};

extern crate ordered_float;
use self::ordered_float::OrderedFloat;

use parser::EXTENV;

type ArgsTypes = Vec<Type>;

fn retrieve_var_name(e: &Closure) -> Option<String> {
    if let &Closure::Var(ref name) = e {
        Some(name.clone())
    } else {
        None
    }
}

// TODO: MUST FIX THIS DIRTY CODE!!

pub fn h(f: FuncDef, p: &mut HashMap<String, Vec<ArgsTypes>>) -> Vec<FuncDef> {
    let (name, ty) = f.name;
    let args_tys_list = if let Some(args_tys_list) = p.get(&name).cloned() {
        args_tys_list
    } else {
        // the function named 'name' is not a polymorphic function.
        return vec![];
    };

    let mut fundefs = Vec::new();
    for args_tys in args_tys_list {
        let mut env = HashMap::new();
        // TODO: should make a func for mangling.
        let mut name_mangled = name.clone() + "#";
        let mut actual_params = vec![];
        for (actual_ty, &(ref param_name, _)) in args_tys.iter().zip(&f.params.clone()) {
            env.insert(param_name.clone(), actual_ty.clone());
            actual_params.push((param_name.clone(), actual_ty.clone()));
            name_mangled += actual_ty.clone().to_string().as_str();
        }
        let (body, ret_ty) = g(*f.body.clone(), &env, p);
        fundefs.push(FuncDef {
            name: (
                name_mangled,
                Type::Func(
                    actual_params.iter().map(|x| x.1.clone()).collect(),
                    Box::new(ret_ty),
                ),
            ),
            params: actual_params,
            formal_fv: f.formal_fv.clone(),
            body: Box::new(body),
        });
    }
    fundefs
}

pub fn g(
    e: Closure,
    env: &HashMap<String, Type>,
    p: &mut HashMap<String, Vec<ArgsTypes>>,
) -> (Closure, Type) {
    macro_rules! g_seq {
        ($es:expr) => ({
            let mut tys = Vec::new();
            for e in $es { tys.push(g(e, env, p)); }
            tys 
        });
    }
    match e {
        Closure::Unit => (e, Type::Unit),
        Closure::Bool(_) => (e, Type::Bool),
        Closure::Int(_) => (e, Type::Int),
        Closure::Float(_) => (e, Type::Float),
        Closure::Var(ref name) => {
            if let Some(ty) = env.get(name.as_str()).cloned() {
                (e.clone(), ty)
            } else {
                panic!()
            }
        }
        Closure::IntBinaryOp(op, lhs, rhs) => {
            (
                Closure::IntBinaryOp(op, Box::new(g(*lhs, env, p).0), Box::new(g(*rhs, env, p).0)),
                Type::Int,
            )
        }
        Closure::LetExpr((name, ty), expr, body) => {
            let mut new_env = env.clone();
            let (expr, expr_ty) = g(*expr, &env, p);
            let ty = if ty.contained_var() { expr_ty } else { ty };
            new_env.insert(name.clone(), ty.clone());
            let (body, body_ty) = g(*body, &new_env, p);
            (
                Closure::LetExpr((name, ty), Box::new(expr), Box::new(body)),
                body_ty,
            )
        }
        Closure::AppCls(callee, args) |
        Closure::AppDir(callee, args) => {
            let name = retrieve_var_name(&*callee).unwrap();
            if EXTENV.lock().unwrap().contains_key(name.as_str()) {
                let args2 = g_seq!(args.clone());
                let mut args = vec![];
                for (e, a) in args2.clone() {
                    args.push(e);
                }
                return (Closure::AppCls(callee, args), Type::Unit);
            }
            let args = g_seq!(args.clone());
            let mut name_mangled = name.clone() + "#";
            let mut args_expr = vec![];
            let mut args_ty = vec![];
            for (e, t) in args.clone() {
                name_mangled += t.to_string().as_str();
                args_expr.push(e);
                args_ty.push(t);
            }
            (*p.entry(name).or_insert(vec![])).push(args_ty);
            (
                Closure::AppCls(Box::new(Closure::Var(name_mangled)), args_expr),
                Type::Unit,
            )
        }
        _ => panic!(),
    }
}

pub fn f(e: Prog) -> Prog {
    let Prog(fundef, expr) = e.clone();
    let mut new_fundef = Vec::new();
    let mut poly_map = HashMap::new();
    let expr = g(expr, &HashMap::new(), &mut poly_map).0;

    for f in fundef {
        new_fundef.extend(h(f, &mut poly_map).iter().cloned());
    }
    Prog(new_fundef, expr)
}

impl Type {
    fn contained_var(&self) -> bool {
        macro_rules! seq {
            ($es:expr) => ({
                for e in $es { if e.contained_var() { return true } }
            });
        }
        match self {
            &Type::Unit | &Type::Char | &Type::Int | &Type::Float => false,
            &Type::Func(ref params_ty, ref ret_ty) => {
                seq!(params_ty);
                ret_ty.contained_var()
            }
            &Type::Var(_) => true,
            _ => panic!(format!("{:?}", self)),
        }
    }
}

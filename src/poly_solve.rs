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

pub fn h(f: FuncDef, p: &mut HashMap<String, Vec<ArgsTypes>>) -> Vec<FuncDef> {
    let mut fundefs = Vec::new();
    let (name, ty) = f.name;
    let poly_args = p.get(&name).unwrap().clone();
    for pargs in poly_args {
        let mut env = HashMap::new();
        let mut name_mangled = name.clone() + "_";
        let mut actual_params = vec![];
        for (actual_ty, &(ref param_name, _)) in pargs.iter().zip(&f.params.clone()) {
            env.insert(param_name.clone(), actual_ty.clone());
            actual_params.push((param_name.clone(), actual_ty.clone()));
            name_mangled += actual_ty.clone().to_string().as_str();
        }
        let (body, ret_ty) = g(*f.body.clone(), &env, p);
        fundefs.push(FuncDef {
            name: (name_mangled, ret_ty),
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
            for e in $es { tys.push(g(e, env, p).1); }
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
        Closure::AppCls(callee, args) |
        Closure::AppDir(callee, args) => {
            let args_ty = g_seq!(args.clone());
            let name = retrieve_var_name(&*callee).unwrap();
            let mut name_mangled = name.clone() + "_";
            for a in args_ty.clone() {
                name_mangled += a.to_string().as_str();
            }
            (*p.entry(name).or_insert(vec![])).push(args_ty);
            (
                Closure::AppCls(Box::new(Closure::Var(name_mangled)), args),
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

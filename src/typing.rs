use std::boxed::Box;
use std::collections::HashMap;

use node::{NodeKind, FuncDef};
use id;

use parser::EXTENV;

#[derive(Clone, PartialEq, Eq, Hash)]
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

// TODO: following functions should return Result<>

pub fn infer_unify(t1: &Type, t2: &Type) {
    match (t1, t2) {
        (&Type::Unit, &Type::Unit) => (),
        (&Type::Bool, &Type::Bool) => (),
        (&Type::Char, &Type::Char) => (),
        (&Type::Int, &Type::Int) => (),
        (&Type::Float, &Type::Float) => (),
        (&Type::Func(ref t1p, ref t1r), &Type::Func(ref t2p, ref t2r)) => {
            if t1p.len() != t2p.len() {
                panic!()
            }
            for (a, b) in t1p.iter().zip(t2p.iter()) {
                infer_unify(a, b)
            }
            infer_unify(t1r, t2r)
        }
        (&Type::Var(i1), &Type::Var(i2)) if i1 == i2 => (),
        (&Type::Var(ref i1), _) => {}
        (_, &Type::Var(_)) => infer_unify(t2, t1),
        // TODO: implement more types
        _ => panic!(),
    }
}

fn infer_sub(
    node: &NodeKind,
    env: &HashMap<String, Type>,
    idgen: &mut id::IdGen,
) -> (NodeKind, Type) {
    macro_rules! infer_seq { ($es:expr) => ({
            let mut tys = Vec::new();
            for e in $es.iter() { tys.push(infer_sub(e, env, idgen).1); }
            tys 
        });
    }
    macro_rules! var_n { ($t:expr) => ({
            if let &Type::Var(n) = $t { n } else { panic!() }
        });
    }

    match *node {
        NodeKind::Unit => (node.clone(), Type::Unit),
        NodeKind::Bool(_) => (node.clone(), Type::Bool),
        NodeKind::Int(_) => (node.clone(), Type::Int),
        NodeKind::Float(_) => (node.clone(), Type::Float),
        NodeKind::Ident(ref name) => {
            if let Some(t) = env.get(name.as_str()).cloned() {
                (NodeKind::Ident(name.clone()), t)
            } else if let Some(t) = EXTENV.lock().unwrap().get(name.as_str()).cloned() {
                (NodeKind::Ident(name.clone()), t)
            } else {
                panic!("not found");
            }
        }
        NodeKind::Call(ref callee, ref args) => {
            let (_, functy) = infer_sub(callee, env, idgen);
            let (ty, params_tys) = if let Type::Func(params_tys, ty) = functy {
                (ty, params_tys)
            } else {
                panic!()
            };
            let args_tys = infer_seq!(args);
            let infered_ty = if let Type::Var(ty_n) = *ty {
                let mut tyenv = HashMap::new();
                for (param_ty, arg_ty) in params_tys.iter().zip(args_tys.iter()) {
                    tyenv.insert(var_n!(param_ty), arg_ty.clone());
                }
                tyenv.get(&ty_n).cloned().unwrap()
            } else {
                *ty
            };
            (node.clone(), infered_ty)
        }
        NodeKind::LetExpr((ref name, ref ty), ref expr, ref body) => {
            let mut newenv = env.clone();
            let infered_expr = infer_sub(expr, env, idgen);
            newenv.insert(name.clone(), infered_expr.1.clone());
            let infered_body = infer_sub(body, &newenv, idgen);
            (
                NodeKind::LetExpr(
                    (name.clone(), infered_expr.1),
                    Box::new(infered_expr.0.clone()),
                    Box::new(infered_body.0.clone()),
                ),
                infered_body.1,
            )
        }
        NodeKind::LetDef((ref name, ref ty), ref expr) => {
            let infered_expr = infer_sub(expr, env, idgen);
            EXTENV.lock().unwrap().insert(
                name.clone(),
                infered_expr.1.clone(),
            );
            (
                NodeKind::LetDef(
                    (name.clone(), infered_expr.1),
                    Box::new(infered_expr.0.clone()),
                ),
                Type::Unit,
            )
        }
        NodeKind::LetFuncExpr(ref funcdef, ref expr, ref body) => {
            let mut newenv = env.clone();
            let mut params_tys = vec![];
            for param in &funcdef.params {
                newenv.insert(param.0.clone(), param.1.clone());
                params_tys.push(param.1.clone());
            }
            newenv.insert(
                funcdef.name.0.clone(),
                Type::Func(params_tys, Box::new(idgen.get_type())),
            );
            let infered_expr = infer_sub(expr, &newenv, idgen);

            let mut newenv_body = env.clone();
            newenv_body.insert(
                funcdef.name.0.clone(),
                Type::Func(
                    funcdef.params.iter().map(|p| p.1.clone()).collect(),
                    Box::new(infered_expr.1.clone()),
                ),
            );
            let infered_body = infer_sub(body, &newenv_body, idgen);
            (
                NodeKind::LetFuncExpr(
                    FuncDef {
                        name: (funcdef.name.0.clone(), infered_expr.1.clone()),
                        params: funcdef.params.clone(),
                    },
                    Box::new(infered_expr.0.clone()),
                    Box::new(infered_body.0.clone()),
                ),
                infered_body.1,
            )
        }
        NodeKind::LetFuncDef(ref funcdef, ref expr) => {
            let mut newenv = env.clone();
            let mut params_tys = vec![];
            for param in &funcdef.params {
                newenv.insert(param.0.clone(), param.1.clone());
                params_tys.push(param.1.clone());
            }
            newenv.insert(
                funcdef.name.0.clone(),
                Type::Func(params_tys.clone(), Box::new(idgen.get_type())),
            );
            let infered_expr = infer_sub(expr, &newenv, idgen);
            EXTENV.lock().unwrap().insert(
                funcdef.name.0.clone(),
                Type::Func(params_tys, Box::new(infered_expr.1.clone())),
            );
            (
                NodeKind::LetFuncDef(
                    FuncDef {
                        name: (funcdef.name.0.clone(), infered_expr.1.clone()),
                        params: funcdef.params.clone(),
                    },
                    Box::new(infered_expr.0.clone()),
                ),
                Type::Unit,
            )
        }
        NodeKind::IntUnaryOp(ref op, ref expr) => {
            let infered_expr = infer_sub(expr, env, idgen);
            infer_unify(&infered_expr.1, &Type::Int);
            // assert
            (
                NodeKind::IntUnaryOp(op.clone(), Box::new(infered_expr.0)),
                infered_expr.1,
            )
        }
        NodeKind::FloatUnaryOp(ref op, ref expr) => {
            let infered_expr = infer_sub(expr, env, idgen);
            infer_unify(&infered_expr.1, &Type::Float);
            // assert
            (
                NodeKind::FloatUnaryOp(op.clone(), Box::new(infered_expr.0)),
                infered_expr.1,
            )
        }
        NodeKind::IntBinaryOp(ref op, ref lhs, ref rhs) => {
            let infered_lhs = infer_sub(lhs, env, idgen);
            let infered_rhs = infer_sub(rhs, env, idgen);
            infer_unify(&infered_lhs.1, &Type::Int);
            infer_unify(&infered_rhs.1, &Type::Int);
            (
                NodeKind::IntBinaryOp(
                    op.clone(),
                    Box::new(infered_lhs.0.clone()),
                    Box::new(infered_rhs.0.clone()),
                ),
                Type::Int,
            )
        }
        NodeKind::FloatBinaryOp(ref op, ref lhs, ref rhs) => {
            let infered_lhs = infer_sub(lhs, env, idgen);
            let infered_rhs = infer_sub(rhs, env, idgen);
            infer_unify(&infered_lhs.1, &Type::Float);
            infer_unify(&infered_rhs.1, &Type::Float);
            (
                NodeKind::FloatBinaryOp(
                    op.clone(),
                    Box::new(infered_lhs.0.clone()),
                    Box::new(infered_rhs.0.clone()),
                ),
                Type::Float,
            )
        }
        _ => panic!(),
    }
}

pub fn infer(node: &NodeKind, idgen: &mut id::IdGen) -> (NodeKind, Type) {
    infer_sub(node, &HashMap::new(), idgen)
}

use std::boxed::Box;
use std::collections::{HashMap, HashSet};

use node::{FuncDef, NodeKind};
use id;

use parser::EXTENV;

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Unit,
    Bool,
    Int,
    Float,
    Char,
    Tuple(Vec<Type>),
    Array(Box<Type>),
    Func(Vec<Type>, Box<Type>), // (param types, return type, is type inference complete?)
    Var(usize),                 // id
}

impl Type {
    pub fn to_string(&self) -> String {
        self.to_string_sub(&mut 0, &mut HashMap::new())
    }

    pub fn to_string_sub(&self, i: &mut usize, m: &mut HashMap<usize, usize>) -> String {
        match self {
            &Type::Unit => "unit".to_string(),
            &Type::Bool => "bool".to_string(),
            &Type::Char => "char".to_string(),
            &Type::Int => "int".to_string(),
            &Type::Float => "float".to_string(),
            &Type::Tuple(ref et) => format!(
                "({})",
                et.into_iter()
                    .fold(
                        "".to_string(),
                        |acc, t| acc + t.to_string_sub(i, m).as_str() + " * "
                    )
                    .trim_right_matches(" * ")
            ),
            &Type::Array(ref et) => format!("[{}]", et.to_string_sub(i, m)),
            &Type::Func(ref param_tys, ref ret_ty) => {
                macro_rules! name { ($id:expr) => ( format!("\'{}", m.entry($id).or_insert_with(|| { *i += 1; *i }).clone()) ) };
                format!(
                    "({})",
                    param_tys
                        .into_iter()
                        .fold("".to_string(), |acc, ts| acc + match *ts {
                            Type::Var(id) => name!(id),
                            _ => ts.to_string_sub(i, m),
                        }.as_str()
                            + " -> ") + if let Type::Var(id) = **ret_ty {
                        name!(id)
                    } else {
                        ret_ty.to_string_sub(i, m)
                    }.as_str()
                )
            }
            &Type::Var(id) => format!("var({})", id),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeScheme {
    pub tyvars: Vec<Type>,
    pub body: Type,
}

impl TypeScheme {
    pub fn new(tyvars: Vec<Type>, body: Type) -> TypeScheme {
        TypeScheme {
            tyvars: tyvars,
            body: body,
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
    macro_rules! deref_ty_seq {
        ($seq:expr) => ($seq.iter().map(|x| deref_ty(x, tyenv))
                                  .collect::<Vec<_>>());
    }
    match *ty {
        Type::Func(ref p, ref r) => Type::Func(deref_ty_seq!(p), Box::new(deref_ty(r, tyenv))),
        Type::Tuple(ref ts) => Type::Tuple(deref_ty_seq!(ts)),
        Type::Array(ref t) => Type::Array(Box::new(deref_ty(t, tyenv))),
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

fn deref_term(node: &NodeKind, tyenv: &mut HashMap<usize, Type>) -> NodeKind {
    macro_rules! deref_seq {
        ($ary:expr) => ($ary.iter().map(|x| deref_term(x, tyenv)).collect::<Vec<_>>());
    }
    match *node {
        NodeKind::IntBinaryOp(ref op, ref lhs, ref rhs) => NodeKind::IntBinaryOp(
            op.clone(),
            Box::new(deref_term(&**lhs, tyenv)),
            Box::new(deref_term(&**rhs, tyenv)),
        ),
        NodeKind::FloatBinaryOp(ref op, ref lhs, ref rhs) => NodeKind::FloatBinaryOp(
            op.clone(),
            Box::new(deref_term(&**lhs, tyenv)),
            Box::new(deref_term(&**rhs, tyenv)),
        ),
        NodeKind::CompBinaryOp(ref op, ref lhs, ref rhs) => NodeKind::CompBinaryOp(
            op.clone(),
            Box::new(deref_term(&**lhs, tyenv)),
            Box::new(deref_term(&**rhs, tyenv)),
        ),
        NodeKind::Tuple(ref es) => NodeKind::Tuple(deref_seq!(es)),
        NodeKind::Call(ref e, ref args) => {
            NodeKind::Call(Box::new(deref_term(e, tyenv)), deref_seq!(args))
        }
        NodeKind::LetExpr((ref name, ref ty), ref expr, ref body) => NodeKind::LetExpr(
            (name.clone(), deref_ty(ty, tyenv)),
            Box::new(deref_term(&**expr, tyenv)),
            Box::new(deref_term(&**body, tyenv)),
        ),
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
        NodeKind::LetTupleExpr(ref es, ref expr, ref body) => {
            let es = es.iter()
                .map(|&(ref e, ref t)| (e.clone(), deref_ty(t, tyenv)))
                .collect::<Vec<_>>();
            NodeKind::LetTupleExpr(
                es,
                Box::new(deref_term(expr, tyenv)),
                Box::new(deref_term(body, tyenv)),
            )
        }
        NodeKind::LetDef((ref name, ref ty), ref expr) => NodeKind::LetDef(
            (name.clone(), deref_ty(ty, tyenv)),
            Box::new(deref_term(&**expr, tyenv)),
        ),
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
        NodeKind::IfExpr(ref cond, ref then_, ref else_) => NodeKind::IfExpr(
            Box::new(deref_term(cond, tyenv)),
            Box::new(deref_term(then_, tyenv)),
            Box::new(deref_term(else_, tyenv)),
        ),
        NodeKind::MakeArray(ref e1, ref e2) => NodeKind::MakeArray(
            Box::new(deref_term(e1, tyenv)),
            Box::new(deref_term(e2, tyenv)),
        ),
        NodeKind::Get(ref e1, ref e2) => NodeKind::Get(
            Box::new(deref_term(e1, tyenv)),
            Box::new(deref_term(e2, tyenv)),
        ),
        NodeKind::Put(ref e1, ref e2, ref e3) => NodeKind::Put(
            Box::new(deref_term(e1, tyenv)),
            Box::new(deref_term(e2, tyenv)),
            Box::new(deref_term(e3, tyenv)),
        ),
        _ => node.clone(),
    }
}

fn occur(r1: usize, ty: &Type) -> bool {
    macro_rules! occur_list {
        ($ls:expr) => ($ls.iter().any(|ty| occur(r1, ty)))
    }
    match *ty {
        Type::Func(ref t2s, ref t2) => occur_list!(t2s) || occur(r1, t2),
        Type::Tuple(ref t2s) => occur_list!(t2s),
        Type::Array(ref t2) => occur(r1, t2),
        Type::Var(r2) => r1 == r2,
        _ => false,
    }
}

pub fn unify(t1: &Type, t2: &Type, tyenv: &mut HashMap<usize, Type>) -> Result<(), TypeError> {
    match (t1, t2) {
        (&Type::Unit, &Type::Unit) => Ok(()),
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
        (&Type::Array(ref t1), &Type::Array(ref t2)) => unify(t1, t2, tyenv),
        (&Type::Tuple(ref t1e), &Type::Tuple(ref t2e)) => {
            if t1e.len() != t2e.len() {
                return Err(TypeError::Unify(t1.clone(), t2.clone()));
            }
            for (a, b) in t1e.iter().zip(t2e.iter()) {
                try!(unify(a, b, tyenv));
            }
            Ok(())
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

fn subst(ty: Type, tyenv: &mut HashMap<usize, Type>, map: HashMap<usize, Type>) -> Type {
    macro_rules! seq {
        ($es:expr) => ({
            let mut argtys = Vec::new();
            for e in $es.iter() { argtys.push(subst(e.clone(), tyenv, map.clone())); }
            argtys
        });
    }
    match ty {
        Type::Unit | Type::Bool | Type::Int | Type::Float | Type::Char => ty,
        Type::Func(params, ret) => Type::Func(seq!(params), Box::new(subst(*ret, tyenv, map))),
        Type::Array(et) => Type::Array(Box::new(subst(*et, tyenv, map))),
        Type::Tuple(es) => Type::Tuple(seq!(es)),
        Type::Var(id) => {
            if let Some(t) = map.get(&id).cloned() {
                t
            } else if let Some(t) = tyenv.get(&id).cloned() {
                subst(t, tyenv, map)
            } else {
                ty
            }
        }
    }
}

fn instantiate(
    tyscheme: TypeScheme,
    tyenv: &mut HashMap<usize, Type>,
    idgen: &mut id::IdGen,
) -> Type {
    let mut map = HashMap::new();
    let oldtyvars = tyscheme.tyvars;
    let mut newtyvars = vec![];
    for o in oldtyvars {
        let v = idgen.get_type();
        map.insert(var_n(&o).unwrap(), v.clone());
        newtyvars.push(v);
    }
    let newbodyty = subst(tyscheme.body, tyenv, map);
    newbodyty
}

fn unwrap_var(ty: Type, tyenv: &mut HashMap<usize, Type>, freevars: &mut Vec<Type>) {
    macro_rules! seq {
        ($es:expr) => ({
            for e in $es.iter() { unwrap_var(e.clone(), tyenv, freevars) }
        });
    }

    match ty {
        Type::Unit | Type::Bool | Type::Int | Type::Float | Type::Char => (),
        Type::Func(params, ret) => {
            seq!(params);
            unwrap_var(*ret, tyenv, freevars)
        }
        Type::Array(et) => unwrap_var(*et, tyenv, freevars),
        Type::Tuple(es) => seq!(es),
        Type::Var(_) => freevars.push(ty.clone()),
    }
}

fn generalize(
    ty: Type,
    env: &HashMap<String, TypeScheme>,
    tyenv: &mut HashMap<usize, Type>,
) -> TypeScheme {
    pub fn subtract<T>(mut lhs: Vec<T>, rhs: &Vec<T>) -> Vec<T>
    where
        T: Clone + Eq + ::std::hash::Hash,
    {
        let mut s = HashSet::new();
        for i in rhs {
            s.insert(i);
        }
        lhs.retain(|i| !s.contains(i));
        lhs
    }

    let ty = deref_ty(&ty, tyenv);
    let mut ty_tyvars = vec![];
    unwrap_var(ty.clone(), tyenv, &mut ty_tyvars);
    // println!("cratepoly >> {:?}", ty_tyvars);
    let mut new_env = Vec::new();
    for (_key, tyscheme) in env {
        let body = deref_ty(&tyscheme.body, tyenv);
        let mut body_tyvars = Vec::new();
        unwrap_var(body.clone(), tyenv, &mut body_tyvars);
        new_env.extend(subtract(body_tyvars, &tyscheme.tyvars));
    }
    let newone = subtract(ty_tyvars, &new_env);
    // println!("generalize >> {:?}", newone);
    TypeScheme::new(newone, ty)
}

fn var_n(ty: &Type) -> Option<usize> {
    if let &Type::Var(n) = ty {
        Some(n)
    } else {
        None
    }
}

pub fn g(
    node: &NodeKind,
    env: &HashMap<String, TypeScheme>,
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
        NodeKind::Unit => Ok(Type::Unit),
        NodeKind::Bool(_) => Ok(Type::Bool),
        NodeKind::Int(_) => Ok(Type::Int),
        NodeKind::Float(_) => Ok(Type::Float),
        NodeKind::Ident(ref name) => {
            if let Some(t) = env.get(name).cloned() {
                Ok(instantiate(t, tyenv, idgen))
            } else if let Some(t) = EXTENV.lock().unwrap().get(name).cloned() {
                Ok(instantiate(t, tyenv, idgen))
            } else {
                println!("{}", name);
                panic!("TODO: implement")
            }
        }
        NodeKind::Tuple(ref es) => Ok(Type::Tuple(g_seq!(es))),
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
        NodeKind::CompBinaryOp(_, ref lhs, ref rhs) => {
            let a = try!(g(lhs, env, tyenv, idgen));
            let b = try!(g(rhs, env, tyenv, idgen));
            try!(unify(&a, &b, tyenv));
            // println!("comp {:?}", tyenv);
            Ok(Type::Bool)
        }
        NodeKind::Call(ref callee, ref args) => {
            let ty = idgen.get_type();
            let callee_ty = try!(g(callee, env, tyenv, idgen));
            let functy = Type::Func(g_seq!(args), Box::new(ty.clone()));
            // println!("call: {:?}", callee_ty);
            // println!("      {:?}", functy);
            try!(unify(&callee_ty, &functy, tyenv));
            Ok(ty)
        }
        NodeKind::LetExpr((ref name, ref ty), ref expr, ref body) => {
            let t = try!(g(expr, env, tyenv, idgen));
            try!(unify(&t, ty, tyenv));
            let p = generalize(t, env, tyenv);
            let mut newenv = env.clone();
            newenv.insert(name.clone(), p);
            g(body, &newenv, tyenv, idgen)
        }
        NodeKind::LetFuncExpr(ref funcdef, ref expr, ref body) => {
            let (name, ty) = funcdef.name.clone();
            let params = &funcdef.params;
            let mut newenv = env.clone();
            newenv.insert(name.clone(), TypeScheme::new(vec![], ty.clone()));
            let mut newenv_body = newenv.clone();
            for &(ref x, ref t) in params.iter() {
                newenv_body.insert(x.to_string(), TypeScheme::new(vec![], t.clone()));
            }
            let newty = Type::Func(
                params.iter().map(|p| p.1.clone()).collect::<Vec<_>>(),
                Box::new(try!(g(expr, &newenv_body, tyenv, idgen))),
            );
            try!(unify(&ty, &newty, tyenv));
            // println!("complete functy: {:?}", newty);
            newenv.insert(name.clone(), generalize(newty, env, tyenv));
            g(body, &newenv, tyenv, idgen)
        }
        NodeKind::LetTupleExpr(ref es, ref expr, ref body) => {
            try!(unify(
                &try!(g(expr, &env, tyenv, idgen)),
                &Type::Tuple(es.iter().map(|e| e.1.clone()).collect::<Vec<_>>(),),
                tyenv,
            ));
            let mut newenv = env.clone();
            for &(ref x, ref t) in es.iter() {
                newenv.insert(x.to_string(), TypeScheme::new(vec![], t.clone()));
            }
            g(body, &newenv, tyenv, idgen)
        }
        NodeKind::LetDef((ref name, ref ty), ref expr) => {
            try!(unify(&try!(g(expr, env, tyenv, idgen)), ty, tyenv));
            let t = generalize(ty.clone(), env, tyenv);
            EXTENV.lock().unwrap().insert(name.clone(), t);
            Ok(Type::Unit)
        }
        NodeKind::LetFuncDef(ref funcdef, ref expr) => {
            let (name, ty) = funcdef.name.clone();
            let params = &funcdef.params;
            let mut newenv = env.clone();
            newenv.insert(name.clone(), TypeScheme::new(vec![], ty.clone()));
            let mut newenv_body = newenv.clone();
            for &(ref x, ref t) in params.iter() {
                newenv_body.insert(x.to_string(), TypeScheme::new(vec![], t.clone()));
            }
            let newty = deref_ty(
                &Type::Func(
                    params.iter().map(|p| p.1.clone()).collect::<Vec<_>>(),
                    Box::new(try!(g(expr, &newenv_body, tyenv, idgen))),
                ),
                tyenv,
            );
            try!(unify(&ty, &newty, tyenv));
            EXTENV
                .lock()
                .unwrap()
                .insert(name.clone(), generalize(newty, env, tyenv));
            Ok(Type::Unit)
        }
        NodeKind::IfExpr(ref cond, ref then_, ref else_) => {
            try!(unify(&try!(g(cond, env, tyenv, idgen)), &Type::Bool, tyenv));
            let t = try!(g(then_, env, tyenv, idgen));
            let e = try!(g(else_, env, tyenv, idgen));
            try!(unify(&t, &e, tyenv));
            Ok(t)
        }
        NodeKind::MakeArray(ref e1, ref e2) => {
            try!(unify(&try!(g(e1, env, tyenv, idgen)), &Type::Int, tyenv));
            let t = try!(g(e2, env, tyenv, idgen));
            Ok(Type::Array(Box::new(t)))
        }
        NodeKind::Get(ref e1, ref e2) => {
            let t = idgen.get_type();
            try!(unify(
                &try!(g(e1, env, tyenv, idgen)),
                &Type::Array(Box::new(t.clone())),
                tyenv
            ));
            try!(unify(&try!(g(e2, env, tyenv, idgen)), &Type::Int, tyenv));
            Ok(t)
        }
        NodeKind::Put(ref e1, ref e2, ref e3) => {
            let t = try!(g(e3, env, tyenv, idgen));
            try!(unify(
                &try!(g(e1, env, tyenv, idgen)),
                &Type::Array(Box::new(t)),
                tyenv
            ));
            try!(unify(&try!(g(e2, env, tyenv, idgen)), &Type::Int, tyenv));
            Ok(Type::Unit)
        }
        _ => panic!(),
    }
}

pub fn f(node: &NodeKind, tyenv: &mut HashMap<usize, Type>, idgen: &mut id::IdGen) -> NodeKind {
    let _infered_ty = g(node, &HashMap::new(), tyenv, idgen);
    // TODO: infered_ty == Unit
    deref_term(node, tyenv)
}

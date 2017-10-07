use std::boxed::Box;
use typing;

#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Bool(bool),
    Int(i32),
    Float(f64),
    Ident(String),
    Call(Box<NodeKind>, Vec<NodeKind>),
    FuncDef((String, typing::Type), Vec<(String, typing::Type)>), // name, params
    LetExpr((String, typing::Type), Box<NodeKind>, Box<NodeKind>), // (name, ty), bound expr, body
    LetFuncExpr(FuncDef, Box<NodeKind>, Box<NodeKind>), // (name, ty), bound expr, body
    LetDef((String, typing::Type), Box<NodeKind>), // name, bound expr
    LetFuncDef(FuncDef, Box<NodeKind>), // name, bound expr
    UnaryOp(UnaryOps, Box<NodeKind>),
    BinaryOp(BinOps, Box<NodeKind>, Box<NodeKind>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncDef {
    pub name: (String, typing::Type),
    pub params: Vec<(String, typing::Type)>,
}

impl NodeKind {
    pub fn get_ident_name(self) -> Option<String> {
        match self {
            NodeKind::Ident(ident) => Some(ident),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOps {
    IAdd,
    FAdd,
    ISub,
    FSub,
    IMul,
    FMul,
    IDiv,
    FDiv,
    IMod,
}

pub fn str_to_binop(opstr: &str) -> BinOps {
    match opstr {
        "+" => BinOps::IAdd,
        "+." => BinOps::FAdd,
        "-" => BinOps::ISub,
        "-." => BinOps::FSub,
        "*" => BinOps::IMul,
        "*." => BinOps::FMul,
        "/" => BinOps::IDiv,
        "/." => BinOps::FDiv,
        _ => BinOps::IAdd,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOps {
    INeg,
    FNeg,
}

pub fn str_to_unaryop(opstr: &str) -> UnaryOps {
    match opstr {
        "-" => UnaryOps::INeg,
        "-." => UnaryOps::FNeg,
        _ => panic!(),
    }
}

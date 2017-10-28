use std::boxed::Box;
use typing;

#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Unit,
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
    IntUnaryOp(UnaryOps, Box<NodeKind>),
    FloatUnaryOp(UnaryOps, Box<NodeKind>),
    IntBinaryOp(BinOps, Box<NodeKind>, Box<NodeKind>),
    FloatBinaryOp(BinOps, Box<NodeKind>, Box<NodeKind>),
    CompBinaryOp(CompBinOps, Box<NodeKind>, Box<NodeKind>),
    IfExpr(Box<NodeKind>, Box<NodeKind>, Box<NodeKind>), // cond, then, else
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

// return (op, is_int_op)
pub fn str_to_binop(opstr: &str) -> (BinOps, bool) {
    match opstr {
        "+" => (BinOps::IAdd, true),
        "+." => (BinOps::FAdd, false),
        "-" => (BinOps::ISub, true),
        "-." => (BinOps::FSub, false),
        "*" => (BinOps::IMul, true),
        "*." => (BinOps::FMul, false),
        "/" => (BinOps::IDiv, true),
        "/." => (BinOps::FDiv, false),
        _ => (BinOps::IAdd, true),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompBinOps {
    SEq,
    SNe,
    PEq,
    PNe,
    Lt,
    Gt,
    Le,
    Ge,
}

pub fn str_to_comp_binop(opstr: &str) -> CompBinOps {
    match opstr {
        "=" => CompBinOps::SEq,
        "<>" => CompBinOps::SNe,
        "==" => CompBinOps::PEq,
        "!=" => CompBinOps::PNe,
        "<" => CompBinOps::Lt,
        ">" => CompBinOps::Gt,
        "<=" => CompBinOps::Le,
        ">=" => CompBinOps::Ge,
        _ => CompBinOps::SEq,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOps {
    INeg,
    FNeg,
}

pub fn str_to_unaryop(opstr: &str) -> (UnaryOps, bool) {
    match opstr {
        "-" => (UnaryOps::INeg, true),
        "-." => (UnaryOps::FNeg, false),
        _ => panic!(),
    }
}

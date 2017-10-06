use std::boxed::Box;

#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Bool(bool),
    Int(i32),
    Float(f64),
    Ident(String),
    Call(Box<NodeKind>, Vec<NodeKind>),
    UnaryOp(UnaryOps, Box<NodeKind>),
    BinaryOp(BinOps, Box<NodeKind>, Box<NodeKind>),
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

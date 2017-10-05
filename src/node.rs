use std::boxed::Box;

#[derive(Debug, Clone)]
pub enum NodeKind {
    Int(i32),
    Float(f64),
    BinaryOp(BinOps, Box<NodeKind>, Box<NodeKind>),
}

#[derive(Debug, Clone)]
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

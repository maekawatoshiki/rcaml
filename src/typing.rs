use std::boxed::Box;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Unit,
    Int,
    Float,
    Char,
    Func(Box<[Type]>, Box<Type>), // (param types, return type)
    Var(Option<Box<Type>>),
}

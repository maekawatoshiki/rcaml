pub mod parser;
pub mod node;
pub mod typing;

#[macro_use]
extern crate nom;

#[link(name = "ffi")]
extern "C" {}

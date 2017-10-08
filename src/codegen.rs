extern crate llvm_sys as llvm;

use self::llvm::core::*;
use self::llvm::prelude::*;

extern crate libc;

use std::ffi::CString;
use std::ptr;
use std::boxed::Box;
use std::collections::{HashMap, hash_map, VecDeque};

use node::NodeKind;
use node;

pub struct CodeGen {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    builder: LLVMBuilderRef,
}

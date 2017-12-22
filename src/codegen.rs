extern crate llvm_sys as llvm;

use self::llvm::core::*;
use self::llvm::prelude::*;

extern crate libc;

use std::ffi::CString;
use std::ptr;
use std::boxed::Box;
use std::collections::HashMap;

use node::{BinOps, CompBinOps};

use closure::{Closure, Prog};
use closure;

use typing::Type;

#[derive(Eq, PartialEq, Hash)]
pub struct ExtFunc {
    llvm_val: LLVMValueRef,
    ty: Type,
}

#[derive(Eq, PartialEq, Clone, Hash)]
pub enum ValKind {
    Load(LLVMValueRef),
    Other(LLVMValueRef),
}

impl ValKind {
    unsafe fn get(&self, builder: LLVMBuilderRef) -> LLVMValueRef {
        match self {
            &ValKind::Load(v) => LLVMBuildLoad(builder, v, CString::new("").unwrap().as_ptr()),
            &ValKind::Other(v) => v,
        }
    }
    unsafe fn retrieve(&self) -> LLVMValueRef {
        match self {
            &ValKind::Load(v) | &ValKind::Other(v) => v,
        }
    }
}

pub struct CodeGen<'a> {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    builder: LLVMBuilderRef,
    exec_engine: llvm::execution_engine::LLVMExecutionEngineRef,
    llvm_main_fun: Option<LLVMValueRef>,
    _tyenv: &'a mut HashMap<usize, Type>,
    ext_funcmap: HashMap<String, ExtFunc>,
    global_varmap: HashMap<String, (Type, LLVMTypeRef, LLVMValueRef)>,
    malloc: LLVMValueRef,
}

#[derive(Debug)]
pub enum CodeGenError {
    Something,
}

type CodeGenResult<T> = Result<T, CodeGenError>;

#[no_mangle]
pub extern "C" fn print_int(i: i32) -> i32 {
    print!("{}", i);
    0
}
#[no_mangle]
pub extern "C" fn print_float(f: f64) -> i32 {
    print!("{}", f);
    0
}
#[no_mangle]
pub extern "C" fn print_newline(_: i32) -> i32 {
    print!("\n");
    0
}
#[no_mangle]
pub extern "C" fn float_of_int(i: i32) -> f64 {
    i as f64
}

unsafe fn cur_bb_has_no_terminator(builder: LLVMBuilderRef) -> bool {
    LLVMIsATerminatorInst(LLVMGetLastInstruction(LLVMGetInsertBlock(builder))) == ptr::null_mut()
}

impl<'a> CodeGen<'a> {
    pub unsafe fn new(tyenv: &'a mut HashMap<usize, Type>) -> Self {
        llvm::execution_engine::LLVMLinkInMCJIT();
        llvm::target::LLVM_InitializeAllTargetMCs();
        llvm::target::LLVM_InitializeNativeTarget();
        llvm::target::LLVM_InitializeNativeAsmPrinter();
        llvm::target::LLVM_InitializeNativeAsmParser();

        let context = LLVMContextCreate();

        let c_mod_name = CString::new("rcaml").unwrap();
        let module = LLVMModuleCreateWithNameInContext(c_mod_name.as_ptr(), context);

        let mut ee = 0 as llvm::execution_engine::LLVMExecutionEngineRef;
        let mut error = 0 as *mut i8;
        if llvm::execution_engine::LLVMCreateExecutionEngineForModule(&mut ee, module, &mut error)
            != 0
        {
            panic!("err");
        }

        let mut ext_funcmap = HashMap::new();
        // initialize standard functions
        // {
        let f_print_int_ty = LLVMFunctionType(
            LLVMInt32Type(),
            vec![LLVMInt32Type()].as_mut_slice().as_mut_ptr(),
            1,
            0,
        );
        let f_print_int = LLVMAddFunction(
            module,
            CString::new("print_int").unwrap().as_ptr(),
            f_print_int_ty,
        );
        ext_funcmap.insert(
            "print_int".to_string(),
            ExtFunc {
                ty: Type::Func(vec![Type::Int], Box::new(Type::Unit)),
                llvm_val: f_print_int,
            },
        );
        llvm::execution_engine::LLVMAddGlobalMapping(
            ee,
            f_print_int,
            print_int as *mut libc::c_void,
        );

        let f_print_float_ty = LLVMFunctionType(
            LLVMInt32Type(),
            vec![LLVMDoubleType()].as_mut_slice().as_mut_ptr(),
            1,
            0,
        );
        let f_print_float = LLVMAddFunction(
            module,
            CString::new("print_float").unwrap().as_ptr(),
            f_print_float_ty,
        );
        ext_funcmap.insert(
            "print_float".to_string(),
            ExtFunc {
                ty: Type::Func(vec![Type::Float], Box::new(Type::Unit)),
                llvm_val: f_print_float,
            },
        );
        llvm::execution_engine::LLVMAddGlobalMapping(
            ee,
            f_print_float,
            print_float as *mut libc::c_void,
        );

        let f_print_newline_ty =
            LLVMFunctionType(LLVMInt32Type(), vec![].as_mut_slice().as_mut_ptr(), 0, 0);
        let f_print_newline = LLVMAddFunction(
            module,
            CString::new("print_newline").unwrap().as_ptr(),
            f_print_newline_ty,
        );
        ext_funcmap.insert(
            "print_newline".to_string(),
            ExtFunc {
                ty: Type::Func(vec![Type::Unit], Box::new(Type::Unit)),
                llvm_val: f_print_newline,
            },
        );
        llvm::execution_engine::LLVMAddGlobalMapping(
            ee,
            f_print_newline,
            print_newline as *mut libc::c_void,
        );

        let f_float_of_int_ty = LLVMFunctionType(
            LLVMDoubleType(),
            vec![LLVMInt32Type()].as_mut_slice().as_mut_ptr(),
            1,
            0,
        );
        let f_float_of_int = LLVMAddFunction(
            module,
            CString::new("float_of_int").unwrap().as_ptr(),
            f_float_of_int_ty,
        );
        ext_funcmap.insert(
            "float_of_int".to_string(),
            ExtFunc {
                ty: Type::Func(vec![Type::Int], Box::new(Type::Float)),
                llvm_val: f_float_of_int,
            },
        );
        llvm::execution_engine::LLVMAddGlobalMapping(
            ee,
            f_float_of_int,
            float_of_int as *mut libc::c_void,
        );

        let f_malloc_ty = LLVMFunctionType(
            LLVMPointerType(LLVMInt8Type(), 0),
            vec![LLVMInt64Type()].as_mut_slice().as_mut_ptr(),
            1,
            0,
        );
        let f_malloc = LLVMAddFunction(
            module,
            CString::new("malloc").unwrap().as_ptr(),
            f_malloc_ty,
        );

        CodeGen {
            context: context,
            module: module,
            builder: LLVMCreateBuilderInContext(context),
            exec_engine: ee,
            llvm_main_fun: None,
            _tyenv: tyenv,
            ext_funcmap: ext_funcmap,
            global_varmap: HashMap::new(),
            malloc: f_malloc,
        }
    }

    pub unsafe fn gen(
        &mut self,
        mod_dump_to_stderr: bool,
        run_module_for_debugging: bool,
        progs: Vec<Prog>,
    ) -> CodeGenResult<LLVMValueRef> {
        let pm = LLVMCreatePassManager();
        llvm::transforms::scalar::LLVMAddTailCallEliminationPass(pm);
        llvm::transforms::scalar::LLVMAddReassociatePass(pm);
        llvm::transforms::scalar::LLVMAddGVNPass(pm);
        llvm::transforms::scalar::LLVMAddInstructionCombiningPass(pm);
        llvm::transforms::scalar::LLVMAddPromoteMemoryToRegisterPass(pm);
        llvm::transforms::scalar::LLVMAddPromoteMemoryToRegisterPass(pm);
        llvm::transforms::scalar::LLVMAddPromoteMemoryToRegisterPass(pm);

        let main_ty = LLVMFunctionType(LLVMInt32Type(), vec![].as_mut_slice().as_mut_ptr(), 0, 0);
        let main = LLVMAddFunction(self.module, CString::new("main").unwrap().as_ptr(), main_ty);
        let bb_entry = LLVMAppendBasicBlock(main, CString::new("entry").unwrap().as_ptr());
        LLVMPositionBuilderAtEnd(self.builder, bb_entry);
        self.llvm_main_fun = Some(main);
        // let mut funcs = Vec::new();

        for Prog(funs, expr) in progs {
            // TODO: consider how to impl poly ty
            let mut env = HashMap::new();
            for fun in funs {
                try!(self.gen_fun(&mut env, &fun));
            }

            try!(self.gen_expr(&env, Some(main), &expr));
        }

        LLVMBuildRet(self.builder, try!(self.gen_int(0)));

        // llvm::analysis::LLVMVerifyModule(
        //     self.module,
        //     llvm::analysis::LLVMVerifierFailureAction::LLVMPrintMessageAction,
        //     0 as *mut *mut i8,
        // );

        // llvm::bit_writer::LLVMWriteBitcodeToFile(
        //     self.module,
        //     CString::new("a.bc").unwrap().as_ptr(),
        // );

        if mod_dump_to_stderr {
            LLVMDumpModule(self.module);
        }

        LLVMRunPassManager(pm, self.module);

        if run_module_for_debugging {
            println!("*** running main ***");
            self.run_module();
            println!("*** end of main ***");
        }

        Ok(ptr::null_mut())
    }

    pub unsafe fn run_module(&mut self) {
        let main = LLVMGetNamedFunction(self.module, CString::new("main").unwrap().as_ptr());
        llvm::execution_engine::LLVMRunFunction(
            self.exec_engine,
            main,
            0,
            vec![].as_mut_slice().as_mut_ptr(),
        );
    }

    unsafe fn gen_fun(
        &mut self,
        env: &mut HashMap<String, ValKind>,
        cls: &closure::FuncDef,
    ) -> CodeGenResult<LLVMValueRef> {
        let tmp_builder = self.builder;
        self.builder = LLVMCreateBuilderInContext(self.context);

        let (ref name, ref fun_ty) = cls.name;
        assert!(match fun_ty {
            &Type::Func(_, _) => true,
            _ => false,
        });

        let llvm_fun_ty = fun_ty.to_llvmty();
        let llvm_fun = LLVMAddFunction(
            self.module,
            CString::new(name.as_str()).unwrap().as_ptr(),
            llvm_fun_ty,
        );

        let bb_entry = LLVMAppendBasicBlock(llvm_fun, CString::new("entry").unwrap().as_ptr());
        LLVMPositionBuilderAtEnd(self.builder, bb_entry);

        if cls.formal_fv.len() > 0 {
            let p = LLVMGetParam(llvm_fun, 0);
            let pty = LLVMPointerType(
                LLVMStructType(
                    cls.formal_fv
                        .iter()
                        .map(|ref x| x.1.to_llvmty())
                        .collect::<Vec<_>>()
                        .as_mut_ptr(),
                    cls.formal_fv.len() as u32,
                    0,
                ),
                0,
            );
            let p = LLVMBuildPointerCast(self.builder, p, pty, CString::new("").unwrap().as_ptr());

            for (i, &(ref param_name, ref param_ty)) in cls.formal_fv.iter().enumerate() {
                let param_val = LLVMBuildStructGEP(
                    self.builder,
                    p,
                    i as u32,
                    CString::new("").unwrap().as_ptr(),
                );
                env.insert(param_name.clone(), ValKind::Load(param_val));
            }
        }

        for (i, &(ref param_name, ref param_ty)) in cls.params.iter().enumerate() {
            //                                         '1' is for free variable
            let param_val = LLVMGetParam(llvm_fun, (i + 1) as u32);
            let var = try!(self.declare_local_var(
                env,
                Some(llvm_fun),
                &param_name,
                param_ty.to_llvmty(),
            ));
            LLVMBuildStore(self.builder, param_val, var);
        }

        env.insert(name.clone(), ValKind::Other(llvm_fun));

        let ret_val = try!(self.gen_expr(&env, Some(llvm_fun), &*cls.body));
        LLVMBuildRet(self.builder, ret_val);

        self.builder = tmp_builder;
        Ok(ptr::null_mut())
    }

    unsafe fn gen_expr(
        &mut self,
        env: &HashMap<String, ValKind>,
        cur_fun: Option<LLVMValueRef>,
        closure: &Closure,
    ) -> CodeGenResult<LLVMValueRef> {
        match closure {
            &Closure::LetExpr((ref name, ref ty), ref expr, ref body) => {
                self.gen_letexpr(env, cur_fun, name, ty, expr, body)
            }
            &Closure::LetTupleExpr(ref xs, ref expr, ref body) => {
                self.gen_lettupleexpr(env, cur_fun, xs, expr, body)
            }
            // LetExpr((String, typing::Type), Box<NodeKind>, Box<NodeKind>), // (name, ty), bound expr, body
            // LetFuncExpr(FuncDef, Box<NodeKind>, Box<NodeKind>), // (name, ty), bound expr, body
            // LetDef((String, typing::Type), Box<NodeKind>), // name, bound expr
            // LetFuncDef(FuncDef, Box<NodeKind>), // name, bound expr

            // TODO: below
            // &NodeKind::LetDef((ref name, ref ty), ref expr) => self.gen_letdef(name, ty, &*expr),
            // &NodeKind::LetFuncDef(ref funcdef, ref expr) => self.gen_letfuncdef(&*funcdef, &*expr),

            // Call(Box<NodeKind>, Vec<NodeKind>),
            &Closure::AppCls(ref callee, ref args) => self.gen_cls(env, cur_fun, &*callee, &*args),
            &Closure::AppDir(ref callee, ref args) => self.gen_dir(env, cur_fun, &*callee, &*args),

            // MakeCls(String, Type, Cls, Box<Closure>),
            &Closure::MakeCls(ref name, ref ty, ref cls, ref body) => {
                self.gen_makecls(env, cur_fun, &*name, &*ty, &*cls, &*body)
            }
            // &NodeKind::Call(ref callee, ref args) => self.gen_call(&*callee, &*args),

            // IntBinaryOp(BinOps, Box<NodeKind>, Box<NodeKind>),
            // &NodeKind::IntBinaryOp(ref op, ref lhs, ref rhs) => {
            //     self.gen_int_binop(op, &*lhs, &*rhs)
            // }
            &Closure::IntBinaryOp(ref op, ref lhs, ref rhs) => {
                self.gen_int_binop(env, cur_fun, &*op, &*lhs, &*rhs)
            }
            &Closure::FloatBinaryOp(ref op, ref lhs, ref rhs) => {
                self.gen_float_binop(env, cur_fun, &*op, &*lhs, &*rhs)
            }
            &Closure::CompBinaryOp(ref op, ref lhs, ref rhs) => {
                self.gen_comp_binop(env, cur_fun, &*op, &*lhs, &*rhs)
            }
            &Closure::If(ref cond, ref then, ref els) => {
                self.gen_if_expr(env, cur_fun, &*cond, &*then, &*els)
            }
            // &NodeKind:: FloatUnaryOp(UnaryOps, Box<NodeKind>)
            &Closure::Var(ref name) => self.gen_var_load(env, name),
            &Closure::Int(ref i) => self.gen_int(*i),
            &Closure::Bool(ref b) => self.gen_bool(*b),
            &Closure::Float(ref f) => self.gen_float(f.into_inner()),
            &Closure::Tuple(ref es) => self.gen_tuple(env, cur_fun, &*es),
            &Closure::Unit => self.gen_int(0), // tmp
            _ => panic!(format!("not implemented {:?}", closure)),
        }
    }

    unsafe fn declare_local_var(
        &mut self,
        env: &mut HashMap<String, ValKind>,
        cur_fun: Option<LLVMValueRef>,
        name: &String,
        ty: LLVMTypeRef,
    ) -> CodeGenResult<LLVMValueRef> {
        let builder = LLVMCreateBuilderInContext(self.context);

        let entry_bb = LLVMGetEntryBasicBlock(cur_fun.unwrap());
        let first_inst = LLVMGetFirstInstruction(entry_bb);
        // let var is always declared at the first of entry block
        if first_inst == ptr::null_mut() {
            LLVMPositionBuilderAtEnd(builder, entry_bb);
        } else {
            LLVMPositionBuilderBefore(builder, first_inst);
        }
        let var = LLVMBuildAlloca(builder, ty, CString::new(name.as_str()).unwrap().as_ptr());
        env.insert(name.to_owned(), ValKind::Load(var));
        Ok(var)
    }

    pub unsafe fn gen_letexpr(
        &mut self,
        env: &HashMap<String, ValKind>,
        cur_fun: Option<LLVMValueRef>,
        name: &String,
        ty: &Type,
        expr: &Closure,
        body: &Closure,
    ) -> CodeGenResult<LLVMValueRef> {
        let mut newenv = env.clone();
        let llvm_expr_val = try!(self.gen_expr(env, cur_fun, expr));
        let var =
            try!(self.declare_local_var(&mut newenv, cur_fun, name, LLVMTypeOf(llvm_expr_val),));
        LLVMBuildStore(self.builder, llvm_expr_val, var);
        self.gen_expr(&newenv, cur_fun, body)
    }

    pub unsafe fn gen_lettupleexpr(
        &mut self,
        env: &HashMap<String, ValKind>,
        cur_fun: Option<LLVMValueRef>,
        xs: &Vec<(String, Type)>,
        expr: &Closure,
        body: &Closure,
    ) -> CodeGenResult<LLVMValueRef> {
        let mut newenv = env.clone();
        let llvm_expr_val = try!(self.gen_expr(env, cur_fun, expr));
        for (i, &(ref name, ref ty)) in xs.iter().enumerate() {
            let llvm_elem_val = try!(self.llvm_struct_elem_extract(llvm_expr_val, i as u32));
            let var =
                try!(
                    self.declare_local_var(&mut newenv, cur_fun, name, LLVMTypeOf(llvm_elem_val),)
                );
            LLVMBuildStore(self.builder, llvm_elem_val, var);
        }
        self.gen_expr(&newenv, cur_fun, body)
    }

    // pub unsafe fn gen_letdef(
    //     &mut self,
    //     name: &String,
    //     ty: &Type,
    //     expr: &NodeKind,
    // ) -> CodeGenResult<LLVMValueRef> {
    //     let llvm_ty = ty.to_llvmty();
    //     let llvm_val = LLVMAddGlobal(
    //         self.module,
    //         llvm_ty,
    //         CString::new(name.as_str()).unwrap().as_ptr(),
    //     );
    //     LLVMSetInitializer(llvm_val, LLVMConstNull(llvm_ty));
    //
    //     self.global_varmap.insert(name.clone(), (
    //         ty.clone(),
    //         llvm_ty,
    //         llvm_val,
    //     ));
    //
    //     let llvm_expr = try!(self.gen_expr(expr));
    //
    //     LLVMBuildStore(self.builder, llvm_expr, llvm_val);
    //
    //     Ok(llvm_val)
    // }
    //
    // pub unsafe fn gen_letfuncdef(
    //     &mut self,
    //     funcdef: &FuncDef,
    //     expr: &NodeKind,
    // ) -> CodeGenResult<LLVMValueRef> {
    //     let (ref func_name, ref func_ret_ty) = funcdef.name;
    //     Ok(ptr::null_mut())
    // }

    unsafe fn llvm_ty_alloc(&mut self, ty: LLVMTypeRef) -> CodeGenResult<LLVMValueRef> {
        let sz = LLVMSizeOf(ty);
        let ty = LLVMPointerType(ty, 0);
        Ok(LLVMBuildPointerCast(
            self.builder,
            LLVMBuildCall(
                self.builder,
                self.malloc,
                vec![sz].as_mut_slice().as_mut_ptr(),
                1,
                CString::new("").unwrap().as_ptr(),
            ),
            ty,
            CString::new("").unwrap().as_ptr(),
        ))
    }
    unsafe fn llvm_struct_alloc(&mut self, vals: Vec<LLVMValueRef>) -> CodeGenResult<LLVMValueRef> {
        let x = try!(
            self.llvm_ty_alloc(LLVMStructType(
                vals.iter()
                    .map(|&v| LLVMTypeOf(v))
                    .collect::<Vec<_>>()
                    .as_mut_slice()
                    .as_mut_ptr(),
                vals.len() as u32,
                0,
            ))
        );
        for (i, val) in vals.iter().enumerate() {
            LLVMBuildStore(
                self.builder,
                *val,
                LLVMBuildStructGEP(
                    self.builder,
                    x,
                    i as u32,
                    CString::new("").unwrap().as_ptr(),
                ),
            );
        }
        Ok(x)
    }
    unsafe fn llvm_struct_elem_load(
        &mut self,
        p: LLVMValueRef,
        idx: u32,
    ) -> CodeGenResult<LLVMValueRef> {
        Ok(LLVMBuildLoad(
            self.builder,
            LLVMBuildStructGEP(self.builder, p, idx, CString::new("").unwrap().as_ptr()),
            CString::new("").unwrap().as_ptr(),
        ))
    }
    unsafe fn llvm_struct_elem_extract(
        &mut self,
        p: LLVMValueRef,
        idx: u32,
    ) -> CodeGenResult<LLVMValueRef> {
        Ok(LLVMBuildExtractValue(
            self.builder,
            p,
            idx,
            CString::new("").unwrap().as_ptr(),
        ))
    }

    unsafe fn make_cls(
        &mut self,
        env: &HashMap<String, ValKind>,
        name: String,
        fun: LLVMValueRef,
        actual_fv: Vec<LLVMValueRef>,
    ) -> CodeGenResult<(HashMap<String, ValKind>, LLVMValueRef)> {
        let actual_fv = if actual_fv.len() == 0 {
            LLVMConstPointerNull(LLVMPointerType(LLVMInt8Type(), 0))
        } else {
            try!(self.llvm_struct_alloc(actual_fv))
        };
        let newcls = try!(self.llvm_struct_alloc(vec![fun, actual_fv]));
        let mut newenv = env.clone();
        newenv.insert(name, ValKind::Other(newcls));
        Ok((newenv, newcls))
    }

    unsafe fn gen_makecls(
        &mut self,
        env: &HashMap<String, ValKind>,
        cur_fun: Option<LLVMValueRef>,
        name: &String,
        _ty: &Type,
        cls: &closure::Cls,
        body: &Closure,
    ) -> CodeGenResult<LLVMValueRef> {
        let fun = try!(self.lookup_var(env, &cls.entry)).retrieve();
        let fv = {
            let mut v = vec![];
            for name in &cls.actual_fv {
                v.push(try!(self.lookup_var(env, name)).get(self.builder))
            }
            v
        };
        let (new_env, _) = try!(self.make_cls(env, name.clone(), fun, fv));
        let x = try!(self.gen_expr(&new_env, cur_fun, body));
        Ok(x)
    }

    unsafe fn gen_cls(
        &mut self,
        env: &HashMap<String, ValKind>,
        cur_fun: Option<LLVMValueRef>,
        callee: &Closure,
        args: &Vec<Closure>,
    ) -> CodeGenResult<LLVMValueRef> {
        let name = if let &Closure::Var(ref name) = callee {
            name
        } else {
            panic!()
        };

        let x = try!(self.lookup_var(env, name)).get(self.builder);
        let fun = try!(self.llvm_struct_elem_load(x, 0));
        let fv = try!(self.llvm_struct_elem_load(x, 1));

        let mut args_val = vec![fv];
        for arg in args {
            let llvm_arg = try!(self.gen_expr(env, cur_fun, &arg));
            args_val.push(llvm_arg);
        }

        // TODO: is this need?
        // if let Some(fun) = self.ext_funcmap.get(name) {
        //     return Ok(LLVMBuildCall(
        //         self.builder,
        //         fun.llvm_val,
        //         args_val.as_mut_slice().as_mut_ptr(),
        //         args_val.len() as u32,
        //         CString::new("").unwrap().as_ptr(),
        //     ));
        // }

        Ok(LLVMBuildCall(
            self.builder,
            fun,
            args_val.as_mut_slice().as_mut_ptr(),
            args_val.len() as u32,
            CString::new("").unwrap().as_ptr(),
        ))
    }

    unsafe fn gen_dir(
        &mut self,
        env: &HashMap<String, ValKind>,
        cur_fun: Option<LLVMValueRef>,
        callee: &Closure,
        args: &Vec<Closure>,
    ) -> CodeGenResult<LLVMValueRef> {
        let name = if let &Closure::Var(ref name) = callee {
            name
        } else {
            panic!()
        };

        let mut args_val = vec![];
        for arg in args {
            args_val.push(try!(self.gen_expr(env, cur_fun, &arg)))
        }

        // TODO: ?
        if let Some(fun) = self.ext_funcmap.get(name) {
            return Ok(LLVMBuildCall(
                self.builder,
                fun.llvm_val,
                args_val.as_mut_slice().as_mut_ptr(),
                args_val.len() as u32,
                CString::new("").unwrap().as_ptr(),
            ));
        }

        let fun = try!(self.lookup_var(env, name)).retrieve();
        args_val.insert(0, LLVMConstNull(LLVMPointerType(LLVMInt8Type(), 0)));

        Ok(LLVMBuildCall(
            self.builder,
            fun,
            args_val.as_mut_slice().as_mut_ptr(),
            args_val.len() as u32,
            CString::new("").unwrap().as_ptr(),
        ))
    }

    unsafe fn gen_int_binop(
        &mut self,
        env: &HashMap<String, ValKind>,
        cur_fun: Option<LLVMValueRef>,
        op: &BinOps,
        lhs: &Closure,
        rhs: &Closure,
    ) -> CodeGenResult<LLVMValueRef> {
        let inst_name = |s: &str| CString::new(s).unwrap().as_ptr();
        let lhs_val = try!(self.gen_expr(env, cur_fun, lhs));
        let rhs_val = try!(self.gen_expr(env, cur_fun, rhs));
        match op {
            &BinOps::IAdd => Ok(LLVMBuildAdd(
                self.builder,
                lhs_val,
                rhs_val,
                inst_name("add"),
            )),
            &BinOps::ISub => Ok(LLVMBuildSub(
                self.builder,
                lhs_val,
                rhs_val,
                inst_name("sub"),
            )),
            &BinOps::IMul => Ok(LLVMBuildMul(
                self.builder,
                lhs_val,
                rhs_val,
                inst_name("mul"),
            )),
            &BinOps::IDiv => Ok(LLVMBuildSDiv(
                self.builder,
                lhs_val,
                rhs_val,
                inst_name("div"),
            )),
            &BinOps::IMod => Ok(LLVMBuildSRem(
                self.builder,
                lhs_val,
                rhs_val,
                inst_name("rem"),
            )),
            _ => panic!("not implemented"),
        }
    }

    unsafe fn gen_float_binop(
        &mut self,
        env: &HashMap<String, ValKind>,
        cur_fun: Option<LLVMValueRef>,
        op: &BinOps,
        lhs: &Closure,
        rhs: &Closure,
    ) -> CodeGenResult<LLVMValueRef> {
        let inst_name = |s: &str| CString::new(s).unwrap().as_ptr();
        let lhs_val = try!(self.gen_expr(env, cur_fun, lhs));
        let rhs_val = try!(self.gen_expr(env, cur_fun, rhs));
        match op {
            &BinOps::FAdd => Ok(LLVMBuildFAdd(
                self.builder,
                lhs_val,
                rhs_val,
                inst_name("fadd"),
            )),
            &BinOps::FSub => Ok(LLVMBuildFSub(
                self.builder,
                lhs_val,
                rhs_val,
                inst_name("fsub"),
            )),
            &BinOps::FMul => Ok(LLVMBuildFMul(
                self.builder,
                lhs_val,
                rhs_val,
                inst_name("fmul"),
            )),
            &BinOps::FDiv => Ok(LLVMBuildFDiv(
                self.builder,
                lhs_val,
                rhs_val,
                inst_name("fdiv"),
            )),
            _ => panic!("not implemented"),
        }
    }

    unsafe fn gen_comp_binop(
        &mut self,
        env: &HashMap<String, ValKind>,
        cur_fun: Option<LLVMValueRef>,
        op: &CompBinOps,
        lhs: &Closure,
        rhs: &Closure,
    ) -> CodeGenResult<LLVMValueRef> {
        let inst_name = |s: &str| CString::new(s).unwrap().as_ptr();
        let lhs_val = try!(self.gen_expr(env, cur_fun, lhs));
        let rhs_val = try!(self.gen_expr(env, cur_fun, rhs));
        match op {
            &CompBinOps::SEq => Ok(LLVMBuildICmp(
                self.builder,
                llvm::LLVMIntPredicate::LLVMIntEQ,
                lhs_val,
                rhs_val,
                inst_name("eq"),
            )),
            &CompBinOps::SNe => Ok(LLVMBuildICmp(
                self.builder,
                llvm::LLVMIntPredicate::LLVMIntNE,
                lhs_val,
                rhs_val,
                inst_name("ne"),
            )),
            &CompBinOps::Lt => Ok(LLVMBuildICmp(
                self.builder,
                llvm::LLVMIntPredicate::LLVMIntSLT,
                lhs_val,
                rhs_val,
                inst_name("lt"),
            )),
            &CompBinOps::Le => Ok(LLVMBuildICmp(
                self.builder,
                llvm::LLVMIntPredicate::LLVMIntSLE,
                lhs_val,
                rhs_val,
                inst_name("le"),
            )),
            &CompBinOps::Gt => Ok(LLVMBuildICmp(
                self.builder,
                llvm::LLVMIntPredicate::LLVMIntSGT,
                lhs_val,
                rhs_val,
                inst_name("gt"),
            )),
            &CompBinOps::Ge => Ok(LLVMBuildICmp(
                self.builder,
                llvm::LLVMIntPredicate::LLVMIntSGE,
                lhs_val,
                rhs_val,
                inst_name("ge"),
            )),
            // TODO: more ops!
            _ => panic!("not supported"),
        }
    }
    unsafe fn gen_if_expr(
        &mut self,
        env: &HashMap<String, ValKind>,
        cur_fun: Option<LLVMValueRef>,
        cond: &Closure,
        then: &Closure,
        els: &Closure,
    ) -> CodeGenResult<LLVMValueRef> {
        macro_rules! tailcall {
            ($closure:expr, $val:expr) => (
                LLVMSetTailCall(
                    $val,
                    match $closure {
                        &Closure::AppCls(_, _) |
                        &Closure::AppDir(_, _) => 1,
                        _ => 0,
                    },
                );
            );
        };
        let cond_val = try!(self.gen_expr(env, cur_fun, cond));

        let fun = cur_fun.unwrap();
        let bb_then = LLVMAppendBasicBlock(fun, CString::new("then").unwrap().as_ptr());
        let bb_else = LLVMAppendBasicBlock(fun, CString::new("else").unwrap().as_ptr());
        let bb_merge = LLVMAppendBasicBlock(fun, CString::new("merge").unwrap().as_ptr());

        LLVMBuildCondBr(self.builder, cond_val, bb_then, bb_else);

        LLVMPositionBuilderAtEnd(self.builder, bb_then);

        let then_val = try!(self.gen_expr(env, cur_fun, then));
        tailcall!(then, then_val);
        // if cur_bb_has_no_terminator(self.builder) {
        let actual_bb_then = LLVMGetInsertBlock(self.builder);
        LLVMBuildBr(self.builder, bb_merge);
        // }

        LLVMPositionBuilderAtEnd(self.builder, bb_else);

        let else_val = try!(self.gen_expr(env, cur_fun, els));
        tailcall!(els, else_val);
        // if cur_bb_has_no_terminator(self.builder) {
        let actual_bb_else = LLVMGetInsertBlock(self.builder);
        LLVMBuildBr(self.builder, bb_merge);
        // }

        LLVMPositionBuilderAtEnd(self.builder, bb_merge);

        let phi = LLVMBuildPhi(
            self.builder,
            LLVMTypeOf(then_val),
            CString::new("phi").unwrap().as_ptr(),
        );
        LLVMAddIncoming(
            phi,
            vec![then_val].as_mut_slice().as_mut_ptr(),
            vec![actual_bb_then].as_mut_slice().as_mut_ptr(),
            1,
        );
        LLVMAddIncoming(
            phi,
            vec![else_val].as_mut_slice().as_mut_ptr(),
            vec![actual_bb_else].as_mut_slice().as_mut_ptr(),
            1,
        );

        Ok(phi)
    }

    unsafe fn lookup_var(
        &mut self,
        env: &HashMap<String, ValKind>,
        name: &String,
    ) -> CodeGenResult<ValKind> {
        if let Some(val) = env.get(name.as_str()) {
            Ok(val.clone())
        } else if let Some(&(ref _ty, _llvmty, val)) = self.global_varmap.get(name.as_str()) {
            Ok(ValKind::Load(val))
        } else {
            panic!(format!("not found variable '{}'", name))
        }
    }

    unsafe fn gen_var_load(
        &mut self,
        env: &HashMap<String, ValKind>,
        name: &String,
    ) -> CodeGenResult<LLVMValueRef> {
        let val = try!(self.lookup_var(env, name));
        Ok(val.get(self.builder))
    }

    unsafe fn gen_tuple(
        &mut self,
        env: &HashMap<String, ValKind>,
        cur_fun: Option<LLVMValueRef>,
        es: &Vec<Closure>,
    ) -> CodeGenResult<LLVMValueRef> {
        let es = {
            let mut v = vec![];
            for e in es.iter().map(|ref e| self.gen_expr(env, cur_fun, e)) {
                v.push(try!(e));
            }
            v
        };
        Ok(LLVMBuildLoad(
            self.builder,
            try!(self.llvm_struct_alloc(es)),
            CString::new("").unwrap().as_ptr(),
        ))
    }

    unsafe fn gen_int(&mut self, i: i32) -> CodeGenResult<LLVMValueRef> {
        Ok(LLVMConstInt(LLVMInt32Type(), i as u64, 0))
    }

    unsafe fn gen_bool(&mut self, b: bool) -> CodeGenResult<LLVMValueRef> {
        Ok(LLVMConstInt(LLVMInt32Type(), if b { 1 } else { 0 }, 0))
    }

    unsafe fn gen_float(&mut self, f: f64) -> CodeGenResult<LLVMValueRef> {
        Ok(LLVMConstReal(LLVMDoubleType(), f))
    }
}

impl Type {
    pub unsafe fn to_llvmty(&self) -> LLVMTypeRef {
        match self {
            &Type::Unit => LLVMInt32Type(),
            &Type::Bool => LLVMInt32Type(),
            &Type::Char => LLVMInt8Type(),
            &Type::Int => LLVMInt32Type(),
            &Type::Float => LLVMDoubleType(),
            &Type::Tuple(ref xs) => LLVMStructType(
                xs.iter()
                    .map(|ref x| x.to_llvmty_sub())
                    .collect::<Vec<_>>()
                    .as_mut_slice()
                    .as_mut_ptr(),
                xs.len() as u32,
                0,
            ),
            &Type::Func(ref params_ty, ref ret_ty) => {
                LLVMFunctionType(
                    ret_ty.to_llvmty_sub(),
                    || -> *mut LLVMTypeRef {
                        let mut param_llvm_types: Vec<LLVMTypeRef> =
                            vec![LLVMPointerType(LLVMInt8Type(), 0)];
                        for param_ty in params_ty {
                            param_llvm_types.push(param_ty.to_llvmty_sub());
                        }
                        param_llvm_types.as_mut_slice().as_mut_ptr()
                    }(),
                    // '1' is for free variable
                    (1 + params_ty.len()) as u32,
                    0,
                )
            }
            _ => panic!(format!("{:?}", self)),
        }
    }
    pub unsafe fn to_llvmty_sub(&self) -> LLVMTypeRef {
        match self {
            &Type::Unit => LLVMInt32Type(),
            &Type::Bool => LLVMInt32Type(),
            &Type::Char => LLVMInt8Type(),
            &Type::Int => LLVMInt32Type(),
            &Type::Float => LLVMDoubleType(),
            &Type::Tuple(ref xs) => LLVMStructType(
                xs.iter()
                    .map(|ref x| x.to_llvmty_sub())
                    .collect::<Vec<_>>()
                    .as_mut_slice()
                    .as_mut_ptr(),
                xs.len() as u32,
                0,
            ),
            &Type::Func(ref params_ty, ref ret_ty) => {
                let fty = LLVMPointerType(
                    LLVMFunctionType(
                        ret_ty.to_llvmty_sub(),
                        || -> *mut LLVMTypeRef {
                            let mut param_llvm_types: Vec<LLVMTypeRef> =
                                vec![LLVMPointerType(LLVMInt8Type(), 0)];
                            for param_ty in params_ty {
                                param_llvm_types.push(param_ty.to_llvmty_sub());
                            }
                            param_llvm_types.as_mut_slice().as_mut_ptr()
                        }(),
                        // '1' is for free variable
                        (1 + params_ty.len()) as u32,
                        0,
                    ),
                    0,
                );
                LLVMPointerType(
                    LLVMStructType(
                        vec![fty, LLVMPointerType(LLVMInt8Type(), 0)]
                            .as_mut_slice()
                            .as_mut_ptr(),
                        2,
                        0,
                    ),
                    0,
                )
            }
            _ => panic!(format!("{:?}", self)),
        }
    }
}

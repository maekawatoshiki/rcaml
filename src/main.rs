extern crate rcaml;
use rcaml::parser;

extern crate clap;
use clap::{Arg, App};

const VERSION_STR: &'static str = env!("CARGO_PKG_VERSION");

fn main() {
    let app = App::new("rcaml")
        .version(VERSION_STR)
        .author("uint256_t")
        .about("rcaml is an OCaml-like implementation in Rust")
        .arg(Arg::with_name("version").short("v").long("version").help(
            "Show version info",
        ))
        .arg(Arg::with_name("FILE")
                .help("Input file")
                // .required(true)
                .index(1))
        .get_matches();

    if app.is_present("version") {
        println!("rcaml {}", VERSION_STR);
        return;
    } else if let Some(filename) = app.value_of("FILE") {
        println!("input filename: {}", filename);
    } else {
        parser::parse_and_show_simple_expr("5 / a3 + 11 * 10");
        parser::parse_and_show_simple_expr("5.2 /. 0.3");
        parser::parse_and_show_simple_expr("a * (b + 3)");
        parser::parse_and_show_simple_expr("-2 * 3");
        parser::parse_and_show_simple_expr("f 1 2");
        parser::parse_and_show_simple_expr("f (g (1 + x) 2)");
        parser::parse_and_show_simple_expr("let x = 1 in x * 2");
        parser::parse_and_show_simple_expr("let f x = x + 1 in f (1 + 2)");

        parser::parse_and_show_module_item("let f x = x * 2;;");

        parser::parse_and_show_module_item("let t = 1, 2 in t");

        parser::parse_and_infer_type("let x = 1 + 2 in x + 1");
        parser::parse_and_infer_type("let f x = if x then 1 else let a = f (x - 1) in a in f 5");
        parser::parse_and_infer_type(
            "let f x = x in let f = f in let a = f 2.3 in let b = f 1 in a",
        );
        parser::parse_and_infer_type(
            "let f x = if x then 1 else let a = f (x - 1) in a + 1 in let b = f 1 in b",
        );
        parser::parse_and_infer_type("let f x = if x + 1 then f x else f x in 1");
        parser::parse_and_infer_type(
            "let f a b = let g c = a = c in g b in let a = f 1 2 in let b = f 1.2 2.3 in b",
        );
        parser::parse_and_infer_type("let id x = x in let f y = id (y id) in let f = f in f");
        parser::parse_and_infer_type("let t = 1, 2.3, true in t");

        parser::parse_and_infer_type_and_closure_conv("let f x = let g y = x + y in g 3 in f 1");
        parser::parse_and_infer_type_and_closure_conv(
            "let fact x = if x = 1 then 1 else x * (fact (x - 1)) in print_int (fact 10)",
        );
        parser::parse_and_infer_type_and_closure_conv("let t = 1, 2.3, false in t");

        // let e = "let f x = x;; f 1;; f 1.3";
        // let e = "let f x = x;; let a = f 1;; let b = f 2.2;;";
        // let e = "let a = 123;; print_int (a + 7);; print_newline ()";
        let e = "let f x = x + 1 in
                    print_int (f 2) ;;
                 print_newline () ;;
                 let g x =
                    let h y = x + y in
                    h x
                 in print_int (g 1) ;;
                 print_newline () ;;
                 let x = 1.2 in
                     let y = 1.11 in
                        print_float (x +. y) ;;
                 print_newline ()
                 let fact x acc = 
                    if x <= 1 then acc
                    else fact (x - 1) (acc + x) in
                 print_int (fact 100 1) ;;
                 print_newline () ;;
                 let fibo x = 
                    if x <= 1 then 1
                    else (fibo (x - 1)) + (fibo (x - 2)) in
                 print_int (fibo 10) ;;
                 print_newline ()";
        parser::parse_module_items(e);
    }
}

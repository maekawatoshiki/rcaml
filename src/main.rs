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

        parser::parse_and_infer_type("let x = 1 + 2 in x + 1");
        parser::parse_and_infer_type("let f x = x in f 1.3");
        parser::parse_and_infer_type("let f x = x in let a = f 1.3 in let b = f 3 in print_int 1");


        // let e = "let f x = x;; f 1;; f 1.3";
        // let e = "let f x = x;; let a = f 1;; let b = f 2.2;;";
        println!("--- following code doesn't run now ---");
        let e = "let f x = x in let a = f 1 in let b = f 2.2 in print_int 1;;";
        // let e = "let a = 123;; print_int a;; print_newline ()";
        println!(">> {}", e);
        let nodes = parser::parse_module_items(e);
        for (i, node) in nodes.iter().enumerate() {
            println!("{}:\t{:?}", i, node);
        }
    }
}

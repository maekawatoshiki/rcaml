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
        parser::parse_simple_expr("5 / a3 + 11 * 10");
        parser::parse_simple_expr("5.2 /. 0.3");
        parser::parse_simple_expr("a * (b + 3)");
        parser::parse_simple_expr("-2 * 3");
        parser::parse_simple_expr("f 1 2");
        parser::parse_simple_expr("f (g (1 + x) 2)");
    }
}

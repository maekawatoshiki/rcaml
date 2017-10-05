extern crate rcaml;

use rcaml::parser;

fn main() {
    parser::parse_simple_expr("5 /2 + 11 * 10");
    parser::parse_simple_expr("5.2 /. 0.3");
}

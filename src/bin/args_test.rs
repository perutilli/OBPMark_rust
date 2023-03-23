use clap::Parser;
use obpmark_rust::benchmark_utils::CommonArgs;

fn main() {
    let args = CommonArgs::parse();
    println!("{:?}", args);
}

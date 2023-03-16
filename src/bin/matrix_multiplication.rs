#![allow(non_snake_case)] // TODO: decide if we want to keep this or not
use clap::Parser;
use core::panic;
use obpmark_rust::matrix::*;
use std::time::Instant;

/// Matrix multiplication benchmark
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Size of the matrix
    #[arg(short, long)]
    size: usize,

    /// Input files names
    #[arg(short, long)]
    input: Option<Vec<String>>,

    /// Prints the output to stdout
    #[arg(short, long, default_value_t = false)]
    output: bool,
}

fn main() {
    let args = Args::parse();
    let size = args.size;
    let A;
    let B;
    let A_1d: Vec<Number>;
    let B_1d: Vec<Number>;

    match args.input {
        Some(_) => {
            panic!("Input files not supported yet");
        }
        None => {
            println!("No input files specified, generating random matrices");
            A = generate_random_matrix(size);
            B = generate_random_matrix(size);
            A_1d = A.iter().flatten().cloned().collect();
            B_1d = B.iter().flatten().cloned().collect();
        }
    }

    let mut C = vec![vec![Number::default(); size]; size];
    let mut C_1d = vec![Number::default(); size * size];

    let now = Instant::now();

    matrix_multiplication(&A, &B, &mut C);

    println!("Elapsed 2d: {:.2?}", now.elapsed());

    let now = Instant::now();

    matrix_multiplication_1d(&A_1d, &B_1d, &mut C_1d, size, size, size);

    println!("Elapsed 1d: {:.2?}", now.elapsed());

    if args.output {
        println!("{}", matrix_to_str(C));
    }
}

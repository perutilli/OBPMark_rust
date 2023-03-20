#![allow(non_snake_case)] // TODO: decide if we want to keep this or not
use clap::Parser;
use core::panic;
use obpmark_rust::{FromRandomSeed, MatMul};
use std::time::Instant;

#[cfg(feature = "1d")]
use obpmark_rust::matrix_1d::Matrix;
#[cfg(feature = "2d")]
use obpmark_rust::matrix_2d::Matrix;
#[cfg(not(any(feature = "1d", feature = "2d", feature = "ndarray")))]
use obpmark_rust::matrix_2d::Matrix;
#[cfg(feature = "ndarray")]
use obpmark_rust::matrix_ndarray::Matrix; // once again for linting reasons

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
    let mut C;

    let seed: u64 = 9453458;

    match args.input {
        Some(_) => {
            panic!("Input files not supported yet");
        }
        None => {
            println!("No input files specified, generating random matrices");
            A = Matrix::from_random_seed(seed, size, size);
            B = Matrix::from_random_seed(seed, size, size);
            // TODO: while not strictly necessary, we would want C to be initialized to 0
            C = Matrix::from_random_seed(seed, size, size);
        }
    }

    let now = Instant::now();

    A.multiply(&B, &mut C).unwrap();

    println!("Elapsed: {:.2?}", now.elapsed());

    if args.output {
        println!("{}", C);
    }
}

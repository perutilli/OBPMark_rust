#![allow(non_snake_case)] // TODO: decide if we want to keep this or not
use clap::Parser;
use core::panic;
use rand::Rng;
use std::time::Instant;

// if not specified, defaults to float
#[cfg(feature = "float")]
type Number = f32;
#[cfg(feature = "double")]
type Number = f64;
#[cfg(feature = "int")]
type Number = i32;
// TODO: maybe find a way to have a default type if none is specified?

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
}

fn main() {
    let args = Args::parse();
    let size = args.size;
    let A;
    let B;

    match args.input {
        Some(_) => {
            panic!("Input files not supported yet");
        }
        None => {
            println!("No input files specified, generating random matrices");
            A = generate_random_matrix(size);
            B = generate_random_matrix(size);
        }
    }

    let now = Instant::now();

    // Code block to measure.
    let C = matrix_multiplication(&A, &B);

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    println!("{:?}", C);
}

// matrices implemented as 2d vectors or as slices? (probably the former??)
// are we ok with the non snake case names given the mathy nature of the code?
pub fn matrix_multiplication(A: &Vec<Vec<Number>>, B: &Vec<Vec<Number>>) -> Vec<Vec<Number>> {
    let n = A.len();
    let m = A[0].len();
    let w = B[0].len();
    assert_eq!(
        m,
        B.len(),
        "Matrix dimensions do not match, impossible to multiply"
    );
    let mut C = vec![vec![Number::default(); w]; n];
    for i in 0..n {
        for j in 0..w {
            for k in 0..m {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    C
}

/// Generates a random square matrix with side length `size`
pub fn generate_random_matrix(size: usize) -> Vec<Vec<Number>> {
    let mut rng = rand::thread_rng();
    let mut matrix = vec![vec![Number::default(); size]; size];
    for i in 0..size {
        for j in 0..size {
            if cfg!(feature = "int") {
                // this is here to avoid overflow when multiplying
                matrix[i][j] = rng.gen_range(0..100);
            } else {
                matrix[i][j] = rng.gen::<Number>();
            }
        }
    }
    matrix
}

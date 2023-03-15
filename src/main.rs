#![allow(non_snake_case)] // TODO: decide if we want to keep this or not
use clap::Parser;
use core::panic;
use rand::Rng;
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
            println!("No input files, generating random matrices");
            A = generate_random_matrix(size);
            B = generate_random_matrix(size);
        }
    }

    let now = Instant::now();

    // Code block to measure.
    let _C = matrix_multiplication(&A, &B);

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    //println!("{:?}", C);
}

type Number = f32; // this can be setup so that it works both with f32 and f64 (and ints as well)

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
    let mut C = vec![vec![0.0; w]; n];
    for i in 0..n {
        for j in 0..w {
            for k in 0..m {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    C
}

pub fn generate_random_matrix(size: usize) -> Vec<Vec<Number>> {
    let mut rng = rand::thread_rng();
    let mut matrix = vec![vec![0.0; size]; size];
    for i in 0..size {
        for j in 0..size {
            matrix[i][j] = rng.gen::<Number>();
        }
    }
    matrix
}

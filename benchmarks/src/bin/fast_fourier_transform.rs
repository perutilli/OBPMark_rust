#![allow(non_snake_case)]
use clap::Parser;
use core::panic;
use obpmark_library::{BaseMatrix, FastFourierTransform};
use std::{path::Path, time::Instant};

use obpmark_library::matrix_1d::Matrix1d as RefMatrix;
use reference_algorithms::fft_function;

use benchmarks::benchmark_utils::{CommonArgs, Implementation, Matrix, Number};

use benchmarks::number;

#[derive(Parser, Debug)]
#[command(about = "FFT benchmark")]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,
}

fn main() {
    let args = Args::parse();

    if args.common.size.count_ones() != 1 {
        // checks that size is power of 2
        panic!("Size must be a power of 2");
    }

    let mut A;

    match args.common.input {
        Some(v) => {
            if v.len() != 1 {
                panic!("Expected 1 input files, got {}", v.len());
            }
            A = Matrix::from_file(Path::new(&v[0]), 1, args.common.size).unwrap();
        }
        None => {
            A = Matrix::from_random_seed(
                args.common.seed,
                1,
                args.common.size,
                number!("-10"),
                number!("10"),
            );
        }
    }

    // this is for validation, since in this case the original matrix is not preserved
    let A_ref = Matrix::new(A.get_data(), 1, args.common.size);

    if args.common.print_input {
        println!("A:");
        println!("{}", A);
    }

    let t0 = Instant::now();

    match args.common.implementation {
        Implementation::Sequential => A.fft(args.common.size >> 1, 0).unwrap(), // the >> 1 is to keep it consistent with the reference implementation
        _ => unimplemented!("Parallel versions not yet implemented"),
    }

    let t1 = Instant::now();

    if args.common.timing {
        println!("Elapsed: {:.2?}", t1 - t0);
    }

    if args.common.output {
        println!("Output:");
        println!("{}", A);
    }

    match args.common.export {
        Some(filename) => {
            // export output
            A.to_file(Path::new(&filename)).unwrap();
        }
        None => (),
    }

    match args.common.verify {
        Some(Some(filename)) => {
            // verify against file
            let A_ref = Matrix::from_file(Path::new(&filename), 1, args.common.size).unwrap();
            if A.get_data() == A_ref.get_data() {
                println!("Verification passed");
            } else {
                println!("Verification failed");
            }
        }
        Some(None) => {
            // verify against reference implementation
            let A_ref = get_ref_result(A_ref, args.common.size);
            if A.get_data() == A_ref.get_data() {
                println!("Verification passed");
            } else {
                println!("Verification failed");
            }
        }
        None => (),
    }
}

fn get_ref_result(A: Matrix, size: usize) -> RefMatrix<Number> {
    let mut A_ref = A.to_c_format();

    // TODO: this is for testing, remove
    let t = Instant::now();
    unsafe {
        fft_function(A_ref.as_mut_ptr(), size >> 1);
    }
    println!("C code: {:.2?}", t.elapsed());

    RefMatrix::new(vec![A_ref], 1, size)
}

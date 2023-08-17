#![allow(non_snake_case)]
use benchmarks::reference_implementations::ccsds_wavelet_transform;
use clap::Parser;
use core::panic;
use obpmark_library::BaseMatrix;
#[cfg(not(feature = "int"))]
use obpmark_library::WaveletTransformFloating;
#[cfg(not(feature = "int"))]
mod constants {
    use benchmarks::benchmark_utils::Number;
    pub const LOW_PASS_FILTER_SIZE: usize = 9;
    pub const HIGH_PASS_FILTER_SIZE: usize = 7;
    pub const LOW_PASS_FILTER: [Number; LOW_PASS_FILTER_SIZE] = [
        0.037828455507,
        -0.023849465020,
        -0.110624404418,
        0.377402855613,
        0.852698679009,
        0.377402855613,
        -0.110624404418,
        -0.023849465020,
        0.037828455507,
    ];
    pub const HIGH_PASS_FILTER: [Number; HIGH_PASS_FILTER_SIZE] = [
        -0.064538882629,
        0.040689417609,
        0.418092273222,
        -0.788485616406,
        0.418092273222,
        0.040689417609,
        -0.064538882629,
    ];
}
#[cfg(feature = "int")]
use obpmark_library::WaveletTransformInteger;
use std::{path::Path, time::Instant};

use benchmarks::benchmark_utils::{CommonArgs, Matrix, Number};
use benchmarks::{number, verify};
use obpmark_library::matrix_1d::Matrix1d as RefMatrix;

#[derive(Parser, Debug)]
#[command(about = "Wavelet transform benchmark")]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,
}

fn main() {
    let args = Args::parse();

    let A;
    let mut B;

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

    if args.common.print_input {
        println!("A:");
        println!("{}", A);
    }

    B = Matrix::zeroes(1, args.common.size);

    let t0 = Instant::now();

    #[cfg(feature = "int")]
    A.wavelet_transform(&mut B, args.common.size / 2).unwrap();
    #[cfg(not(feature = "int"))]
    A.wavelet_transform(
        &mut B,
        args.common.size / 2,
        &constants::LOW_PASS_FILTER,
        constants::LOW_PASS_FILTER_SIZE,
        &constants::HIGH_PASS_FILTER,
        constants::HIGH_PASS_FILTER_SIZE,
    )
    .unwrap();
    let t1 = Instant::now();

    if args.common.timing {
        println!("Elapsed: {:.2?}", t1 - t0);
    }

    if args.common.output {
        println!("Output:");
        println!("{}", B);
    }

    match args.common.export {
        Some(filename) => {
            // export output
            B.to_file(Path::new(&filename)).unwrap();
        }
        None => (),
    }

    match args.common.verify {
        Some(Some(filename)) => {
            // verify against file
            let B_ref = Matrix::from_file(Path::new(&filename), args.common.size, args.common.size)
                .unwrap();
            if B.get_data() == B_ref.get_data() {
                println!("Verification passed");
            } else {
                println!("Verification failed");
            }
        }
        Some(None) => {
            // verify against cpu implementation
            let B_ref = get_ref_result(A, args.common.size);
            verify!(B.get_data(), B_ref.get_data());
        }
        None => (),
    }
}

fn get_ref_result(A: Matrix, size: usize) -> RefMatrix<Number> {
    let A_ref = A.to_c_format();

    let mut B_ref = vec![number!("0"); 1 * size];

    // TODO: this is for testing, remove
    let t = Instant::now();
    unsafe { ccsds_wavelet_transform(A_ref.as_ptr(), B_ref.as_mut_ptr(), size / 2) }
    println!("C code: {:.2?}", t.elapsed());

    RefMatrix::new(vec![B_ref], 1, size)
}

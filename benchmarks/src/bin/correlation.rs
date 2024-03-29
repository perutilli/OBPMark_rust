#![allow(non_snake_case)]
use clap::Parser;
use core::panic;
use obpmark_library::{
    //parallel_traits::ParallelCorrelation,
    rayon_traits::RayonCorrelation,
    BaseMatrix,
    Correlation,
};
use std::{path::Path, time::Instant};

use benchmarks::benchmark_utils::{CommonArgs, Implementation, Matrix, Number};

use reference_algorithms::correlation;

use benchmarks::number;

#[cfg(feature = "float")]
type Output = f32;
#[cfg(feature = "double")]
type Output = f64;
#[cfg(feature = "int")]
type Output = f32;
#[cfg(not(any(feature = "float", feature = "double", feature = "int",)))]
type Output = f32;

#[derive(Parser, Debug)]
#[command(about = "2D correlation benchmark")]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,
}

fn main() {
    let args = Args::parse();

    let A;
    let B;

    match args.common.input {
        Some(v) => {
            if v.len() != 2 {
                panic!("Expected 2 input files, got {}", v.len());
            }
            A = Matrix::from_file(Path::new(&v[0]), args.common.size, args.common.size).unwrap();
            B = Matrix::from_file(Path::new(&v[1]), args.common.size, args.common.size).unwrap();
        }
        None => {
            A = Matrix::from_random_seed(
                args.common.seed,
                args.common.size,
                args.common.size,
                number!("-10"),
                number!("10"),
            );
            B = Matrix::from_random_seed(
                args.common.seed + 10,
                args.common.size,
                args.common.size,
                number!("-10"),
                number!("10"),
            );
        }
    }

    if args.common.print_input {
        println!("A:");
        println!("{}", A);
        println!("B:");
        println!("{}", B);
    }

    let t0 = Instant::now();
    let res;

    match (args.common.nthreads, args.common.implementation) {
        (None, Implementation::Sequential) => res = A.correlation(&B).unwrap(),
        (Some(_), Implementation::Sequential) => {
            panic!("Cannot specify number of threads for sequential implementation")
        }
        (None, Implementation::Rayon) => res = A.rayon_correlate(&B).unwrap(),
        (Some(_), Implementation::Rayon) => {
            /*
            rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads)
                .build_global()
                .unwrap();
            res = A.rayon_correlate(&B).unwrap();
            */
            panic!("Specifying number of threads for rayon is not supported");
        }
        (Some(_n_threads), Implementation::StdParallel) => {
            // res = A.parallel_correlate(&B, n_threads).unwrap()
            panic!("CHANGE THIS")
        }
        (None, Implementation::StdParallel) => {
            panic!("Must specify number of threads for std parallel");
        }
    }

    let t1 = Instant::now();

    if args.common.timing {
        println!("Elapsed: {:.2?}", t1 - t0);
    }

    if args.common.output {
        println!("Correlation = {}", res);
    }

    match args.common.export {
        Some(filename) => {
            // export output
            // TODO: this is a very hacky way to do this, make it better
            obpmark_library::matrix_1d::Matrix1d::<Output>::new(vec![vec![res; 1]; 1], 1, 1)
                .to_file(Path::new(&filename))
                .unwrap();
        }
        None => (),
    }

    match args.common.verify {
        Some(Some(_filename)) => {
            /* verify against file TODO: how do we deal with this?
            let C_ref = Matrix::from_file(Path::new(&filename), args.common.size, args.common.size)
                .unwrap();
            if C.get_data() == C_ref.get_data() {
                println!("Verification passed");
            } else {
                println!("Verification failed");
            }
             */
        }
        Some(None) => {
            // verify against cpu implementation
            let res_ref = get_ref_result(A, B, args.common.size);
            if res - res_ref > 1e-4 || res_ref - res > 1e-4 {
                println!("Verification failed");
            } else {
                println!("Verification passed");
            }
        }
        None => (),
    }
}

fn get_ref_result(A: Matrix, B: Matrix, size: usize) -> Output {
    let A_ref = A.to_c_format();
    let B_ref = B.to_c_format();
    let mut res_ref = 0.0;
    let t0 = Instant::now();
    unsafe {
        correlation(
            A_ref.as_ptr(),
            B_ref.as_ptr(),
            &mut res_ref as *mut Output,
            size,
        );
    }
    let t1 = Instant::now();
    println!("C code: {:.2?}", t1 - t0);
    res_ref
}

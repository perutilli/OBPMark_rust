#![allow(non_snake_case)]
use clap::Parser;
use core::panic;
use obpmark_rust::{BaseMatrix, FastFourierTransform};
use std::{path::Path, time::Instant};

use obpmark_rust::benchmark_utils::{CommonArgs, Matrix, Number};

use obpmark_rust::number;

#[derive(Parser, Debug)]
#[command(about = "Matrix multiplication benchmark")]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,
}

fn main() {
    let args = Args::parse();

    let seed: u64 = 34523459;

    let mut A;

    match args.common.input {
        Some(v) => {
            if v.len() != 1 {
                panic!("Expected 1 input files, got {}", v.len());
            }
            A = Matrix::from_file(Path::new(&v[0]), 1, args.common.size).unwrap();
        }
        None => {
            A = Matrix::from_random_seed(seed, 1, args.common.size, number!("-10"), number!("10"));
        }
    }

    if args.common.print_input {
        println!("A:");
        println!("{}", A);
    }

    let t0 = Instant::now();

    match args.common.parallel {
        1 => A.fft(args.common.size >> 1, 0).unwrap(), // the >> 1 is to keep it consistent with the reference implementation
        _n => unimplemented!("Parallel version not yet implemented"),
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
            let A_ref = Matrix::from_file(Path::new(&filename), args.common.size, args.common.size)
                .unwrap();
            if A.get_data() == A_ref.get_data() {
                println!("Verification passed");
            } else {
                println!("Verification failed");
            }
        }
        Some(None) => {
            // verify against cpu implementation
            todo!("Need to consider what should be the reference implementation/if this even makes sense");
        }
        None => (),
    }
}

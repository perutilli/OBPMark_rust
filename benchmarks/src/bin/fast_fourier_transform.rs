#![allow(non_snake_case)]
use clap::Parser;
use core::panic;
use obpmark_library::{BaseMatrix, FastFourierTransform};
use std::{path::Path, time::Instant};

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
            // verify against cpu implementation
            todo!("Need to consider what should be the reference implementation/if this even makes sense");
        }
        None => (),
    }
}

#![allow(non_snake_case)]
use clap::Parser;
use core::panic;
use obpmark_library::{BaseMatrix, FastFourierTransformWindowed};
use std::{path::Path, time::Instant};

use benchmarks::benchmark_utils::{CommonArgs, Implementation, Matrix, Number};

use benchmarks::number;

#[derive(Parser, Debug)]
#[command(about = "FFT windowed benchmark")]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,

    /// Size of the window
    #[arg(short, long)]
    window: usize,
}

fn main() {
    let args = Args::parse();

    let mut A;
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

    let n_elements_B = ((args.common.size - args.window) + 1) * args.window;
    B = Matrix::zeroes(1, n_elements_B);

    let t0 = Instant::now();

    match args.common.implementation {
        // Note that this call will modify A as well as B
        Implementation::Sequential => A.fftw(args.common.size >> 1, args.window, &mut B).unwrap(),
        _ => unimplemented!("Parallel versions not yet implemented"),
    }

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
            A.to_file(Path::new(&filename)).unwrap();
        }
        None => (),
    }

    match args.common.verify {
        Some(Some(filename)) => {
            // verify against file
            let B_ref = Matrix::from_file(Path::new(&filename), 1, n_elements_B).unwrap();
            if B.get_data() == B_ref.get_data() {
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

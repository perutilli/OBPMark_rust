use clap::Parser;
use obpmark_library::parallel_traits::{
    ParallelConvolution, ParallelLRN, ParallelMatMul, ParallelMaxPooling, ParallelRelu,
    ParallelSoftmax,
};
use obpmark_library::rayon_traits::{
    RayonConvolution, RayonLRN, RayonMatMul, RayonMaxPooling, RayonRelu, RayonSoftmax,
};
use std::path::Path;
use std::time::Instant;

use obpmark_library::{BaseMatrix, Convolution, MatMul, MaxPooling, Padding, Relu, Softmax, LRN};

use benchmarks::benchmark_utils::{CommonArgs, Implementation, Matrix, Number};
use benchmarks::number;

const CIFAR_10_INPUT: usize = 32;
const CIFAR_10_OUTPUT: usize = 10;
const KERNEL_CON_1: usize = 3;
const KERNEL_CON_2: usize = 3;
const STRIDE_1: usize = 2;
const STRIDE_2: usize = 2;
const DENSE_1: usize = 384;
const DENSE_2: usize = 10;

const K: Number = 2.0;
const ALPHA: Number = 10e-4;
const BETA: Number = 0.75;

#[derive(Parser, Debug)]
#[command(about = "Cifar10 benchmark")]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,

    /// Number of images to process (default: 1)
    #[arg(long, default_value_t = 1)]
    pub images: usize,
}

fn main() {
    let args = Args::parse();

    let mut input = Vec::new();

    let kernel_1;
    let kernel_2;

    // NOTE: weights_1 dimensions:
    // dense_layer_1_out = weights_1 * pooling_2_out
    // dense_layer_1_out = DENSE_1 x 1
    // pooling_2_out = (CIFAR_10_INPUT / STRIDE_1 /STRIDE_2) * (CIFAR_10_INPUT / STRIDE_1 /STRIDE_2) x 1
    // => weights_1 = DENSE_1 x (CIFAR_10_INPUT / STRIDE_1 /STRIDE_2) * (CIFAR_10_INPUT / STRIDE_1 /STRIDE_2)
    // this means that pooling_2_out needs to be reshaped
    let weights_1;
    let weights_1_rows = DENSE_1;
    let weights_1_cols =
        (CIFAR_10_INPUT / STRIDE_1 / STRIDE_2) * (CIFAR_10_INPUT / STRIDE_1 / STRIDE_2);

    // NOTE: weights_2 dimensions:
    // dense_layer_2_out = weights_2 * relu_3_out
    // dense_layer_2_out = DENSE_2 x 1
    // relu_3_out = DENSE_1 x 1
    // => weights_2 = DENSE_2 x DENSE_1
    // no need for reshaping
    let weights_2;
    let weights_2_rows = DENSE_2;
    let weights_2_cols = DENSE_1;

    match (args.common.input, args.images) {
        (Some(v), 1) => {
            if v.len() != 2 {
                panic!("Expected 2 input files, got {}", v.len());
            }
            input
                .push(Matrix::from_file(Path::new(&v[0]), CIFAR_10_INPUT, CIFAR_10_INPUT).unwrap());
            kernel_1 = Matrix::from_file(Path::new(&v[1]), KERNEL_CON_1, KERNEL_CON_1).unwrap();
            kernel_2 = Matrix::from_file(Path::new(&v[2]), KERNEL_CON_2, KERNEL_CON_2).unwrap();
            weights_1 =
                Matrix::from_file(Path::new(&v[3]), weights_1_rows, weights_1_cols).unwrap();
            weights_2 =
                Matrix::from_file(Path::new(&v[4]), weights_2_rows, weights_2_cols).unwrap();
        }
        (Some(_), _) => panic!("Input files not supported for multiple images"),
        (None, n) => {
            for i in 0..n {
                input.push(Matrix::from_random_seed(
                    args.common.seed + 5 + i as u64,
                    CIFAR_10_INPUT,
                    CIFAR_10_INPUT,
                    number!("-10"),
                    number!("10"),
                ));
            }
            kernel_1 = Matrix::from_random_seed(
                args.common.seed + 1,
                KERNEL_CON_1,
                KERNEL_CON_1,
                number!("-10"),
                number!("10"),
            );
            kernel_2 = Matrix::from_random_seed(
                args.common.seed + 2,
                KERNEL_CON_2,
                KERNEL_CON_2,
                number!("-10"),
                number!("10"),
            );
            weights_1 = Matrix::from_random_seed(
                args.common.seed + 3,
                weights_1_rows,
                weights_1_cols,
                number!("-10"),
                number!("10"),
            );
            weights_2 = Matrix::from_random_seed(
                args.common.seed + 4,
                weights_2_rows,
                weights_2_cols,
                number!("-10"),
                number!("10"),
            );
        }
    }

    if args.common.print_input {
        for i in 0..args.images {
            println!("input_{}:", i);
            println!("{}", input[i]);
        }
        println!("kernel_1:");
        println!("{}", kernel_1);
        println!("kernel_2:");
        println!("{}", kernel_2);
        println!("weights_1:");
        println!("{}", weights_1);
        println!("weights_2:");
        println!("{}", weights_2);
    }

    // Initialization of intermediate matrices
    let mut conv_1_out = Matrix::zeroes(CIFAR_10_INPUT, CIFAR_10_INPUT);
    let mut relu_1_out = Matrix::zeroes(CIFAR_10_INPUT, CIFAR_10_INPUT);
    let mut pooling_1_out = Matrix::zeroes(CIFAR_10_INPUT / STRIDE_1, CIFAR_10_INPUT / STRIDE_1);
    let mut lrn_1_out = Matrix::zeroes(CIFAR_10_INPUT / STRIDE_1, CIFAR_10_INPUT / STRIDE_1);
    let mut conv_2_out = Matrix::zeroes(CIFAR_10_INPUT / STRIDE_1, CIFAR_10_INPUT / STRIDE_1);
    let mut relu_2_out = Matrix::zeroes(CIFAR_10_INPUT / STRIDE_1, CIFAR_10_INPUT / STRIDE_1);
    let mut lrn_2_out = Matrix::zeroes(CIFAR_10_INPUT / STRIDE_1, CIFAR_10_INPUT / STRIDE_1);
    let mut pooling_2_out = Matrix::zeroes(
        CIFAR_10_INPUT / STRIDE_1 / STRIDE_2,
        CIFAR_10_INPUT / STRIDE_1 / STRIDE_2,
    );
    let mut dense_layer_1_out = Matrix::zeroes(DENSE_1, 1);
    let mut relu_3_out = Matrix::zeroes(DENSE_1, 1);
    let mut dense_layer_2_out = Matrix::zeroes(DENSE_2, 1);
    let mut relu_4_out = Matrix::zeroes(DENSE_2, 1);

    // Initialization of output matrix
    let mut output = Matrix::zeroes(CIFAR_10_OUTPUT, 1);

    let t0 = Instant::now();

    // Run the benchmark
    cifar_10_multiple(
        &input,
        args.images,
        &kernel_1,
        &kernel_2,
        &weights_1,
        &weights_2,
        &mut output,
        &mut conv_1_out,
        &mut relu_1_out,
        &mut pooling_1_out,
        &mut lrn_1_out,
        &mut conv_2_out,
        &mut relu_2_out,
        &mut lrn_2_out,
        &mut pooling_2_out,
        &mut dense_layer_1_out,
        &mut relu_3_out,
        &mut dense_layer_2_out,
        &mut relu_4_out,
        STRIDE_1,
        STRIDE_2,
        args.common.implementation,
        args.common.nthreads.unwrap_or(4),
    );

    let t1 = Instant::now();

    if args.common.timing {
        println!("Elapsed: {:.2?}", t1 - t0);
    }

    if args.common.output {
        println!("Output:");
        println!("{}", output);
    }

    match args.common.export {
        Some(filename) => {
            // export output
            output.to_file(Path::new(&filename)).unwrap();
        }
        None => (),
    }

    // TODO: add verification code
}

fn cifar_10_multiple(
    input: &Vec<Matrix>,
    n_images: usize,
    kernel_1: &Matrix,
    kernel_2: &Matrix,
    weights_1: &Matrix,
    weights_2: &Matrix,
    output: &mut Matrix,
    conv_1_out: &mut Matrix,
    relu_1_out: &mut Matrix,
    pool_1_out: &mut Matrix,
    lrn_1_out: &mut Matrix,
    conv_2_out: &mut Matrix,
    relu_2_out: &mut Matrix,
    lrn_2_out: &mut Matrix,
    pool_2_out: &mut Matrix,
    dense_layer_1_out: &mut Matrix,
    relu_3_out: &mut Matrix,
    dense_layer_2_out: &mut Matrix,
    relu_4_out: &mut Matrix,
    stride_1_size: usize,
    stride_2_size: usize,
    implementation: Implementation,
    n_threads: usize,
) {
    for i in 0..n_images {
        match implementation {
            Implementation::Sequential => {
                cifar_10(
                    &input[i],
                    kernel_1,
                    kernel_2,
                    weights_1,
                    weights_2,
                    output,
                    conv_1_out,
                    relu_1_out,
                    pool_1_out,
                    lrn_1_out,
                    conv_2_out,
                    relu_2_out,
                    lrn_2_out,
                    pool_2_out,
                    dense_layer_1_out,
                    relu_3_out,
                    dense_layer_2_out,
                    relu_4_out,
                    stride_1_size,
                    stride_2_size,
                );
            }
            Implementation::Rayon => {
                cifar_10_rayon(
                    &input[i],
                    kernel_1,
                    kernel_2,
                    weights_1,
                    weights_2,
                    output,
                    conv_1_out,
                    relu_1_out,
                    pool_1_out,
                    lrn_1_out,
                    conv_2_out,
                    relu_2_out,
                    lrn_2_out,
                    pool_2_out,
                    dense_layer_1_out,
                    relu_3_out,
                    dense_layer_2_out,
                    relu_4_out,
                    stride_1_size,
                    stride_2_size,
                );
            }
            Implementation::StdParallel => {
                cifar_10_parallel(
                    &input[i],
                    kernel_1,
                    kernel_2,
                    weights_1,
                    weights_2,
                    output,
                    conv_1_out,
                    relu_1_out,
                    pool_1_out,
                    lrn_1_out,
                    conv_2_out,
                    relu_2_out,
                    lrn_2_out,
                    pool_2_out,
                    dense_layer_1_out,
                    relu_3_out,
                    dense_layer_2_out,
                    relu_4_out,
                    stride_1_size,
                    stride_2_size,
                    n_threads,
                );
            }
        }
        cifar_10(
            &input[i],
            kernel_1,
            kernel_2,
            weights_1,
            weights_2,
            output,
            conv_1_out,
            relu_1_out,
            pool_1_out,
            lrn_1_out,
            conv_2_out,
            relu_2_out,
            lrn_2_out,
            pool_2_out,
            dense_layer_1_out,
            relu_3_out,
            dense_layer_2_out,
            relu_4_out,
            stride_1_size,
            stride_2_size,
        );
    }
}

fn cifar_10(
    input: &Matrix,
    kernel_1: &Matrix,
    kernel_2: &Matrix,
    weights_1: &Matrix,
    weights_2: &Matrix,
    output: &mut Matrix,
    conv_1_out: &mut Matrix,
    relu_1_out: &mut Matrix,
    pool_1_out: &mut Matrix,
    lrn_1_out: &mut Matrix,
    conv_2_out: &mut Matrix,
    relu_2_out: &mut Matrix,
    lrn_2_out: &mut Matrix,
    pool_2_out: &mut Matrix,
    dense_layer_1_out: &mut Matrix,
    relu_3_out: &mut Matrix,
    dense_layer_2_out: &mut Matrix,
    relu_4_out: &mut Matrix,
    stride_1_size: usize,
    stride_2_size: usize,
) {
    // 1-1 Convolution
    input
        .convolute(&kernel_1, Padding::Zeroes, conv_1_out)
        .unwrap();

    // 1-2 Activation (ReLU)
    conv_1_out.relu(relu_1_out).unwrap();

    // 1-3 Max pooling
    relu_1_out
        .max_pooling(pool_1_out, stride_1_size, stride_1_size)
        .unwrap();

    // 1-4 Normalization (LRN)
    pool_1_out.lrn(lrn_1_out, ALPHA, BETA, K).unwrap();

    // 2-1 Convolution
    lrn_1_out
        .convolute(kernel_2, Padding::Zeroes, conv_2_out)
        .unwrap();

    // 2-2 Activation (ReLU)
    conv_2_out.relu(relu_2_out).unwrap();

    // 2-3 Normalization (LRN)
    relu_2_out.lrn(lrn_2_out, ALPHA, BETA, K).unwrap();

    // 2-4 Max pooling
    lrn_2_out
        .max_pooling(pool_2_out, stride_2_size, stride_2_size)
        .unwrap();

    // Reshape
    pool_2_out
        .reshape(
            (CIFAR_10_INPUT / STRIDE_1 / STRIDE_2) * (CIFAR_10_INPUT / STRIDE_1 / STRIDE_2),
            1,
        )
        .unwrap();

    // Dense layer 1
    weights_1.multiply(pool_2_out, dense_layer_1_out).unwrap();

    // Activation (ReLU)
    dense_layer_1_out.relu(relu_3_out).unwrap();

    // Dense layer 2
    weights_2.multiply(relu_3_out, dense_layer_2_out).unwrap();

    // Activation (ReLU)
    dense_layer_2_out.relu(relu_4_out).unwrap();

    // Softmax
    relu_4_out.softmax(output).unwrap();

    // Reshape for next iteration
    // TODO: this is very unoptimal for 2d matrices, it might be better to
    //       straight up reallocate pool_2_out
    pool_2_out
        .reshape(
            CIFAR_10_INPUT / STRIDE_1 / STRIDE_2,
            CIFAR_10_INPUT / STRIDE_1 / STRIDE_2,
        )
        .unwrap();
}

fn cifar_10_rayon(
    input: &Matrix,
    kernel_1: &Matrix,
    kernel_2: &Matrix,
    weights_1: &Matrix,
    weights_2: &Matrix,
    output: &mut Matrix,
    conv_1_out: &mut Matrix,
    relu_1_out: &mut Matrix,
    pool_1_out: &mut Matrix,
    lrn_1_out: &mut Matrix,
    conv_2_out: &mut Matrix,
    relu_2_out: &mut Matrix,
    lrn_2_out: &mut Matrix,
    pool_2_out: &mut Matrix,
    dense_layer_1_out: &mut Matrix,
    relu_3_out: &mut Matrix,
    dense_layer_2_out: &mut Matrix,
    relu_4_out: &mut Matrix,
    stride_1_size: usize,
    stride_2_size: usize,
) {
    // 1-1 Convolution
    input
        .rayon_convolute(&kernel_1, Padding::Zeroes, conv_1_out)
        .unwrap();

    // 1-2 Activation (ReLU)
    conv_1_out.rayon_relu(relu_1_out).unwrap();

    // 1-3 Max pooling
    relu_1_out
        .rayon_max_pooling(pool_1_out, stride_1_size, stride_1_size)
        .unwrap();

    // 1-4 Normalization (LRN)
    pool_1_out.rayon_lrn(lrn_1_out, ALPHA, BETA, K).unwrap();

    // 2-1 Convolution
    lrn_1_out
        .rayon_convolute(kernel_2, Padding::Zeroes, conv_2_out)
        .unwrap();

    // 2-2 Activation (ReLU)
    conv_2_out.rayon_relu(relu_2_out).unwrap();

    // 2-3 Normalization (LRN)
    relu_2_out.rayon_lrn(lrn_2_out, ALPHA, BETA, K).unwrap();

    // 2-4 Max pooling
    lrn_2_out
        .rayon_max_pooling(pool_2_out, stride_2_size, stride_2_size)
        .unwrap();

    // Reshape
    pool_2_out
        .reshape(
            (CIFAR_10_INPUT / STRIDE_1 / STRIDE_2) * (CIFAR_10_INPUT / STRIDE_1 / STRIDE_2),
            1,
        )
        .unwrap();

    // Dense layer 1
    weights_1
        .rayon_multiply(pool_2_out, dense_layer_1_out)
        .unwrap();

    // Activation (ReLU)
    dense_layer_1_out.rayon_relu(relu_3_out).unwrap();

    // Dense layer 2
    weights_2
        .rayon_multiply(relu_3_out, dense_layer_2_out)
        .unwrap();

    // Activation (ReLU)
    dense_layer_2_out.rayon_relu(relu_4_out).unwrap();

    // Softmax
    relu_4_out.rayon_softmax(output).unwrap();

    // Reshape for next iteration
    // TODO: this is very unoptimal for 2d matrices, it might be better to
    //       straight up reallocate pool_2_out
    pool_2_out
        .reshape(
            CIFAR_10_INPUT / STRIDE_1 / STRIDE_2,
            CIFAR_10_INPUT / STRIDE_1 / STRIDE_2,
        )
        .unwrap();
}

fn cifar_10_parallel(
    input: &Matrix,
    kernel_1: &Matrix,
    kernel_2: &Matrix,
    weights_1: &Matrix,
    weights_2: &Matrix,
    output: &mut Matrix,
    conv_1_out: &mut Matrix,
    relu_1_out: &mut Matrix,
    pool_1_out: &mut Matrix,
    lrn_1_out: &mut Matrix,
    conv_2_out: &mut Matrix,
    relu_2_out: &mut Matrix,
    lrn_2_out: &mut Matrix,
    pool_2_out: &mut Matrix,
    dense_layer_1_out: &mut Matrix,
    relu_3_out: &mut Matrix,
    dense_layer_2_out: &mut Matrix,
    relu_4_out: &mut Matrix,
    stride_1_size: usize,
    stride_2_size: usize,
    n_threads: usize,
) {
    // 1-1 Convolution
    input
        .parallel_convolute(&kernel_1, Padding::Zeroes, conv_1_out, n_threads)
        .unwrap();

    // 1-2 Activation (ReLU)
    conv_1_out.parallel_relu(relu_1_out, n_threads).unwrap();

    // 1-3 Max pooling
    relu_1_out
        .parallel_max_pooling(pool_1_out, stride_1_size, stride_1_size, n_threads)
        .unwrap();

    // 1-4 Normalization (LRN)
    pool_1_out
        .parallel_lrn(lrn_1_out, ALPHA, BETA, K, n_threads)
        .unwrap();

    // 2-1 Convolution
    lrn_1_out
        .parallel_convolute(kernel_2, Padding::Zeroes, conv_2_out, n_threads)
        .unwrap();

    // 2-2 Activation (ReLU)
    conv_2_out.parallel_relu(relu_2_out, n_threads).unwrap();

    // 2-3 Normalization (LRN)
    relu_2_out
        .parallel_lrn(lrn_2_out, ALPHA, BETA, K, n_threads)
        .unwrap();

    // 2-4 Max pooling
    lrn_2_out
        .parallel_max_pooling(pool_2_out, stride_2_size, stride_2_size, n_threads)
        .unwrap();

    // Reshape
    pool_2_out
        .reshape(
            (CIFAR_10_INPUT / STRIDE_1 / STRIDE_2) * (CIFAR_10_INPUT / STRIDE_1 / STRIDE_2),
            1,
        )
        .unwrap();

    // Dense layer 1
    weights_1
        .parallel_multiply(pool_2_out, dense_layer_1_out, n_threads)
        .unwrap();

    // Activation (ReLU)
    dense_layer_1_out
        .parallel_relu(relu_3_out, n_threads)
        .unwrap();

    // Dense layer 2
    weights_2
        .parallel_multiply(relu_3_out, dense_layer_2_out, n_threads)
        .unwrap();

    // Activation (ReLU)
    dense_layer_2_out
        .parallel_relu(relu_4_out, n_threads)
        .unwrap();

    // Softmax
    relu_4_out.parallel_softmax(output, n_threads).unwrap();

    // Reshape for next iteration
    // TODO: this is very unoptimal for 2d matrices, it might be better to
    //       straight up reallocate pool_2_out
    pool_2_out
        .reshape(
            CIFAR_10_INPUT / STRIDE_1 / STRIDE_2,
            CIFAR_10_INPUT / STRIDE_1 / STRIDE_2,
        )
        .unwrap();
}

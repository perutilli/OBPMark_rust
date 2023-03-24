## Current tasks
* [x] Accept non common arguments: test clap(flatten)
* [ ] Improve benchmark template to reflect changes
* [ ] Implement all benchmarks in their naive (1d and 2d vector) form (original names):
    - [ ] cifar_10
    - [ ] cifar_10_multiple
    - [ ] convolution_2D_bench
    - [ ] correlation_2D
    - [ ] fast_fourier_transform_2D_bench
    - [ ] fast_fourier_transform_bench
    - [ ] fast_fourier_transform_window_bench
    - [ ] finite_impulse_response_bench
    - [ ] LNR_bench
    - [x] matrix_multiplication_bench
    - [ ] matrix_multiplication_bench_fp16
    - [ ] matrix_multiplication_tensor_bench
    - [ ] max_pooling_bench
    - [ ] memory_bandwidth_bench
    - [x] relu_bench
    - [ ] softmax_bench TODO: implement it for int, need to wait for Leonidas
    - [ ] wavelet_transform
* [x] Implement from_file and to_file for Matrix types
* [ ] Create unit tests for unit testable functions (this will need a list)

## Questions for Leonidas
* Softmax does not seem to work for int in the C cpu implementation, here is the output I get:
    ```bash
    ./bin/softmax_cpu_int_16 -s 10 -o
    Using device: Generic device
    0 -2 -2 -2 -2 -2 -2 0 -2 0 
    -2 -2 -2 -2 -2 -2 -2 0 0 -2 
    -2 -2 0 -2 -2 -2 -2 0 -2 -2 
    0 -2 -2 0 -2 -2 -2 0 0 -2 
    0 -2 -2 -2 -2 -2 0 0 -2 -2 
    0 -2 -2 -2 -2 -2 -2 0 -2 0 
    -2 0 -2 -2 0 -2 0 -2 -2 -2 
    0 -2 -2 -2 -2 0 0 -2 0 0 
    -2 -2 -2 -2 -2 0 -2 -2 -2 -2 
    -2 -2 0 0 -2 -2 -2 -2 -2 -2 
    ```
    What would be the correct output? If the output has to be int then wouldn't this just be argmax?


## Future improvements
* Improve verification so that the benchmark contains only the code that is unique to it
* Improve ouput formatting to take config like number of digits after the decimal point etc (https://doc.rust-lang.org/std/fmt/)
* Many things that are pub now might be better as pub(crate) probably
* Make a macro or something so that the 1d indexing can be written as the 2d one
* Sadly I think we will need the matrices to be generic (OBPMark repo benchmarks would be easier if we do it this way)
* Add information to file errors
* Maybe move input outside of CommonArgs so it can require the exact number of files in depending on the benchmark
* Move seed to config or something
* Make non common but required arguments more prominent (e.g. stride in max pooling)

### RNGs
Each benchmark could potentially want a different rng, however I don't know that this is worth the effort.  
For now the rng will generate random numbers between -100 and 100, regardless of the type of the matrix.  
Using an offset to the seed to generate different matrices in the mat mul benchmark.

### Ndarray correcteness
As (maybe) expected, the ndarray fails verification for float and double. However sometimes it does pass using epsilon of 1e-4 (same used in the original benchmark). Why does this happen?  
Also how come it is so fast -> Quite sure it uses BLAS or something similar, we need to be sure of this.   
<mark> I think we might want to abandon ndarray at least for the moment</mark>, it might not be mantained anymore. I do want to come back to it, but it seems a bit too opaque of a package. 

## Standardizing arguments

### Common arguments
List of all common arguments for all benchmarks, and ones that could be common if not already present, not necessarily with the same exact behaviour as the original benchmark:
* size (-s, --size): size of matrix or matrices (this can be common for the current implementations since they are all square matrices and if there are 2 they are the same size)
* export (-e, --export): export the results of the output in hexadecimal format (right now it says also verification, for the moment I would leave that out) (this would be -g at the moment)
* verify (-v, --verify): verify the output against the reference cpu implementation (for now it will be against the Rust's 2d vector implementation, later we might want this to be against the original C implementation) or against the file specified
* NOT CONSIDERED -g: not considered at the moment
* output (-o, --output): print the results to stdout in a human readable format that is not suitable for verification (hence it is probably fine to keep it for stdout)
* timing (-t, --timing): print the timing of the execution: in the gpu version there are 3 times (copy to gpu, kernel execution, copy back to cpu), for now we only have one
* NOT CONSIDERED -c, -C: having only one time, this does not seem useful
* input (-i, --input): pass input data (using the hex format)
* NOT RELEVANT -d: not relevant for cpu benchmarks
* help (-h, --help): print help information, taken care of by clap
We can put this in a clap struct in a module "benchmark_utils" or something like that, and then we can import it in all the benchmarks.  
Then we should think about a macro or something that will take care of the common arguments (this might be a bit tricky, leaving it for later)

#### Potential additions
* print inputs

### Arguments specific to the benchmarks
TODO:

## Benchmark organization
Is any combination of the arguments valid? 
Only problematic one I can think of is verify and input, it would make more sense to have input have only the input matrix and verify in two flavours:
* with empty value -> verify against the cpu implementation
* with value -> verify against the file with name provided  
  
This way all arguments are valid in any combination  
Should we check valid values for different arguments before running the benchmark?
* size and input are fine because they happen before the benchmark (TODO: note the interaction of size and input!)
* timing, output and export are fine because they are flags -> cannot have invalid values
* verify could have an invalid filename -> what do we do if we fail to open the verification file? two options:
    - skip verification, signal the error to the user
    - verify against the cpu implementation, signal the error to the user

```Rust
...
fn main() {
    let args = CommonArgs::parse();

    match args.input {
        Some(v) => {
            // read input from file
            if v.len() != EXPECTED {
                panic!("Expected EXP input files, got {}", v.len());
            }
            // read the matrix/matrices
        },
        None => {
            // generate input
        }
    }

    // run the benchmark
    // timing regardless of -t

    if args.timing {
        // print timing
    }

    if args.output {
        // print output
    }

    match args.export {
        Some(filename) => {
            // export output
        }
        None => (),
    }

    match args.verify {
        Some(Some(filename)) => {
            // verify against file
        },
        Some(None) => {
            // verify against cpu implementation
        },
        None => (),
    }
}

```


## Original arguments

```C
void print_usage(const char * appName)
{
	printf("Usage: %s -s Size [-v] [-e] [-o] [-t] [-d] [-i input_file_A_MATRIX input_file_B_MATRIX] \n", appName);
	printf(" -s Size : set size of x and y of matrices A and B with Size \n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -d: selects GPU\n");
	printf(" -h: print help information\n");
}

void print_usage(const char * appName)
{
	printf("Usage: %s -s Size [-v] [-e] [-o] [-t] [-d] [-i input_file_A_MATRIX input_file_B_MATRIX] \n", appName);
	printf(" -s Size : set size of x and y of matrices A and B with Size \n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -C: prints the timing in csv format with timestamp\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -d: selects GPU\n");
	printf(" -f: mutes all print\n");
	printf(" -h: print help information\n");
}

void print_usage(const char * appName)
{
	printf("Usage: %s -s Size -k [-v] [-e] [-o] [-t] [-d] [-i input_file_A_MATRIX input_file_B_MATRIX] \n", appName);
	printf(" -s Size : set size of x and y of matrices A and B with Size \n");
	printf(" -k: size of the kernel\n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -C: prints the timing in csv format with timestamp\n");
	printf(" -q: prints input values\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -d: selects GPU\n");
	printf(" -f: mutes all print\n");
	printf(" -h: print help information\n");
}
void print_usage(const char * appName)
{
	printf("Usage: %s -s Size [-w] [-v] [-e] [-o] [-t] [-c] [-d] [-i input_file_A_MATRIX ] \n", appName);
	printf(" -s Size : set size of furier transform power of 2 \n");
	printf(" -w: window size power of 2 and smaller than size\n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -C: prints the timing in csv format with timestamp\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -q: prints input\n");
	printf(" -d: selects GPU\n");
	printf(" -h: print help information\n");
}

void print_usage(const char * appName)
{
	printf("Usage: %s -s Size [-v] [-e] [-o] [-t] [-d] [-i input_file_A_MATRIX input_file_B_MATRIX] \n", appName);
	printf(" -s Size : set size of x and y of matrices A and B with Size \n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -C: prints the timing in csv format with timestamp\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -d: selects GPU\n");
	printf(" -f: mutes all print\n");
	printf(" -h: print help information\n");
}

void print_usage(const char * appName)
{
	printf("Usage: %s -s Size [-w] [-v] [-e] [-o] [-t] [-c] [-d] [-i input_file_A_MATRIX ] \n", appName);
	printf(" -s Size : set size of furier transform power of 2 \n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -C: prints the timing in csv format with timestamp\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -q: prints input\n");
	printf(" -d: selects GPU\n");
	printf(" -h: print help information\n");
}

void print_usage(const char * appName)
{
	printf("Usage: %s -s Size -k [-v] [-e] [-o] [-t] [-d] [-i input_file_A_MATRIX input_file_B_MATRIX] \n", appName);
	printf(" -s Size : set size of x and y of matrix A and matrix B\n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -C: prints the timing in csv format with timestamp\n");
	printf(" -q: prints input values\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -d: selects GPU\n");
	printf(" -f: mutes all print\n");
	printf(" -h: print help information\n");
}

void print_usage(const char * appName)
{
	printf("Usage: %s -s Size -k [-v] [-e] [-o] [-t] [-d] [-i input_file_A_MATRIX input_file_B_MATRIX] \n", appName);
	printf(" -s Size : set size of x and y of matrices A and B with Size \n");
	printf(" -k: size of the kernel\n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -C: prints the timing in csv format with timestamp\n");
	printf(" -q: prints input values\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -d: selects GPU\n");
	printf(" -f: mutes all print\n");
	printf(" -h: print help information\n");
}
```







## General notes
<mark>Remember to use the `--release` flag when compiling the benchmarks if testing the performance!!!</mark>

Cargo features are not supposed to be used the way we are using them, however there does not seem to be a straight forward way to achieve the same functionality. The question is whether we care enough about the size of the binary to keep using conditional compilation (in this "wrong" way, we would enforce proper features values like so https://internals.rust-lang.org/t/mutually-exclusive-feature-flags/8601) or if we can just deal with this stuff with clap.  
I believe that we do care about the size of the binary, given that the code will run on embedded devices.

There exist blas and lapack ports to rust (https://github.com/stainless-steel/blas, https://github.com/stainless-steel/lapack), however they are autogenerated and all the calls are unsafe (which might not be fixable).  
While this is likely not a problem since this libraries have existed for decades, this might go against the idea of having a safe language, making them, in my opinion, not suitable for this benchmark.  
These are some probably more suitable alternatives:
* nalgebra: Linear algebra library (https://www.nalgebra.org/docs/)
* ndarray: provides an N-dimensional container for general elements and for numerics. (https://docs.rs/ndarray)
* rulinalg a linear algebra library written in Rust that doesn’t require heavy external dependencies. (https://github.com/AtheMathmo/rulinalg/) -> apparently not maintained anymore 
 
Is this even the direction we want to go? I think we might want to implement this ourselves, in which case though we should probably have this be a separate library, so that the code is usable not exclusively for this benchmark.

How can I check correctness??

Our structure could be the following:

`lib.rs`
```rust
mod matrix_1d;
mod matrix_2d;
mod matrix_ndarray;

// error enum definition

trait MatMul {
    fn mat_mul(&self, other: &Self, res: &mut Self) -> Result<(), Error>;
}
```

`matrix_1d.rs`
```rust
pub struct Matrix {
    data: Vec<Number>,
    rows: usize,
    cols: usize,
}

impl MatMul for Matrix {
    fn mat_mul(&self, other: &Self, res: &mut Self) -> Result<(), Error> {
        // do stuff
    }
}
```

`matrix_2d.rs`
```rust
pub struct Matrix {
    data: Vec<Vec<Number>>,
    rows: usize,
    cols: usize,
}
```

`main.rs`
```rust
#[cfg(feature = "matrix_1d")]
use matrix_1d::Matrix;
#[cfg(feature = "matrix_2d")]
use matrix_2d::Matrix;
#[cfg(feature = "matrix_ndarray")]
use matrix_ndarray::Matrix;

```
I think it should check all the boxes:  
- [x] straight forward and localized conditional compilation
- [x] binary size should be the same as if we had a single implementation
- [x] completely transparent to the main function, it only has to use the `Matrix` type
- [x] each benchmark defines a new trait and implements it for the various matrix types
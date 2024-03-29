## Sequential benchmarks
* [ ] Fix problems with fft and fft windowed (they seem to not work anymore)
* [x] Implement all benchmarks in their naive (1d and 2d vector) form (original names):
    - [x] cifar_10
    - [x] cifar_10_multiple
    - [x] convolution_2D_bench
    - [x] correlation_2D
    - [x] NOT AVAILABLE IN SEQ: fast_fourier_transform_2D_bench
    - [x] fast_fourier_transform_bench
    - [x] fast_fourier_transform_window_bench
    - [x] finite_impulse_response_bench => special case of convolution (1d)
    - [x] LNR_bench
    - [x] matrix_multiplication_bench
    - [x] matrix_multiplication_bench_fp16;
    - [x] max_pooling_bench
    - [ ] memory_bandwidth_bench; Do a memcpy (I think)
    - [x] relu_bench
    - [x] softmax_bench TODO: It does not make sense for int, should be enforced at compile time
    - [x] wavelet_transform

## Parallel Benchmarks
**Current parallelization is only good for square matrices, or at least with n_threads rows**. We should deal with this at least in the 1d version. It is not a priority though given that the benchmarks worked only on square matrices.

### Parallel benchmarks tasks (rayon)
* [x] Improve arguments, problem: -p necessary but value ignored
* [ ] Implement all parallel benchmarks with rayon parallel:
    - [ ] cifar_10
    - [ ] cifar_10_multiple
    - [x] convolution_2D_bench
    - [ ] correlation_2D
    - [ ] NOT AVAILABLE IN SEQ: fast_fourier_transform_2D_bench
    - [ ] fast_fourier_transform_bench
    - [ ] fast_fourier_transform_window_bench
    - [x] finite_impulse_response_bench => special case of convolution (1d) 
    - [x] LNR_bench
    - [x] matrix_multiplication_bench
    - [x] matrix_multiplication_bench_fp16; // Note that this did not take any additional development time
    - [x] max_pooling_bench
    - [ ] memory_bandwidth_bench; Does not apply (?)
    - [x] relu_bench
    - [x] softmax_bench TODO: It does not make sense for int, should be enforced at compile time
    - [ ] wavelet_transform

### Parallel benchmarks tasks (naive)
* [x] Decide on a design: this could include some sort of macro that deals with the splitting of the data, maybe even arc cloning and thread scope 
* [ ] Implement all parallel benchmarks in naive form:
    - [ ] cifar_10
    - [ ] cifar_10_multiple
    - [x] convolution_2D_bench => looks quite a bit faster than the rayon version (both in 1d and 2d)
    - [ ] correlation_2D
    - [ ] NOT AVAILABLE IN SEQ: fast_fourier_transform_2D_bench
    - [ ] fast_fourier_transform_bench
    - [ ] fast_fourier_transform_window_bench
    - [x] finite_impulse_response_bench => special case of convolution (1d)
    - [x] LNR_bench
    - [x] matrix_multiplication_bench
    - [x] matrix_multiplication_bench_fp16;
    - [x] max_pooling_bench
    - [ ] memory_bandwidth_bench; Does not apply (?)
    - [x] relu_bench
    - [x] softmax_bench TODO: It does not make sense for int, should be enforced at compile time
    - [ ] wavelet_transform

## Bare metal rust
- [ ] Show proof of concept program running on bare metal (no parallelism or data structures)

## General tasks
* [ ] NEED TO WRITE DOCUMENTATION FOR ALL LIBRARY FUNCTIONS
* [x] Probably move to num_traits (from num)
* [x] Move seed to argument in clap
* [ ] Add information to all errors
* [ ] Improve wavelet transform both benchmark side and library side
* [ ] size in cifar10 doesn't do anything, however it is required by the common arguments, I think we should change this
    - Probably by having a separete macro that expands all the common arguments except size and then using that inside cifar10
* [ ] Standardize position of result inside the calls to the benchmark functions (probaly first parameter)
* [x] Fix formatting for f16 (it does not work with typical formatting)
* [x] Maybe move to workspace organization (https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html)
* [ ] Use OpenMP implementations to test fft, fft windowed and maybe cifar 10
* [x] Moving to generics for lib
    - [x] Fixing problems with verify function (mod benchmark_utils)
    - [x] Basic testing
    - [x] Check all the code to see that it makes sense
    - [ ] Make issue on funty, I think it makes sense to point this out
* [x] Accept non common arguments: test clap(flatten)
* [x] Improve benchmark template to reflect changes
* [ ] Test --input and --export functionalities
* [x] Implement from_file and to_file for Matrix types
* [ ] Test the performance against the original benchmarks
* [ ] Rethink significantly the num traits stuff
  - [ ] It might be good to move some of the more complicated stuff to macros where we cannot easily use traits, avoiding to much headaches in trying to make the traits work (like in fft)
* [ ] Create unit tests for unit testable functions (this will need a list)
* [ ] fft macro could be a single one for both 1d and 2d given that the the matrix always has 1 row
* [x] understand how we should verify our output against matlab for fft
* [x] remove ndarray probably


## Future improvements
* Improve verification so that the benchmark contains only the code that is unique to it
* Improve ouput formatting to take config like number of digits after the decimal point etc (https://doc.rust-lang.org/std/fmt/)
* Many things that are pub now might be better as pub(crate) probably
* Make a macro or something so that the 1d indexing can be written as the 2d one **note the errors that the lack of this caused in our code (commit in the beginning of may)**
* Maybe move input outside of CommonArgs so it can require the exact number of files in depending on the benchmark
* [x] Move seed to config or something (Done: we moved it to the common args and given it a default value)
* Right now we will just panic if --export or --verify provide invalid paths, should handle this better

### Open questions
* How do we do verification in a resonable and automated way?
  * It could make sense to have a script that runs the rust version with different seeds (we would have to add an option to output the matrix(ces)), then having the output and inputs we run the c code with the inputs and have a script compare the outputs (i don't know how automizable this could be)
  * Otherwise ffi would be the move probably, I think this is quite a bit more complicated

### Generics
* We still need a type (`Number`) for the GPU4S which will be known at compile time, probably defined in benchmark_utils
* The matrices have to be generic over a type (let's call it `Num`) which will have a bunch of traits (`Add<Output = Num>`, `Sub`, `Mul`, `Div`, etc)
* This way we have 2 domains:
    - The library one (Matrix, Matrix2d, etc) which is fully generic
    - The GPU4S benchmark one (matrix_multiplication_bench, etc) which is not generic and uses the `Number` type (known at compile time)

### RNGs
Each benchmark could potentially want a different rng, however I don't know that this is worth the effort.  
For now the rng will generate random numbers between -100 and 100, regardless of the type of the matrix. - NOT TRUE ANYMORE (the range is passed as an argument) 
CHECK THIS: Using an offset to the seed to generate different matrices in the mat mul benchmark.

### Ndarray correcteness
As (maybe) expected, the ndarray fails verification for float and double. However sometimes it does pass using epsilon of 1e-4 (same used in the original benchmark). Why does this happen?  
Also how come it is so fast -> Quite sure it uses BLAS or something similar, we need to be sure of this.   
<mark>I think we might want to abandon ndarray at least for the moment</mark>, it might not be mantained anymore. I do want to come back to it, but it seems a bit too opaque of a package. 

## Standardizing arguments

### Common arguments
List of all common arguments for all benchmarks, and ones that could be common if not already present, not necessarily with the same exact behaviour as the original benchmark:
* size (-s, --size): size of matrix or matrices (this can be common for the current implementations since they are all square matrices and if there are 2 they are the same size)
* export (-e, --export): export the results of the output in hexadecimal format (right now it says also verification, for the moment I would leave that out) (this would be -g at the moment)
* verify (-v, --verify): verify the output against the reference cpu implementation (for now it will be against the Rust's 2d vector implementation, later we might want this to be against the original C implementation) or against the file specified
* output (-o, --output): print the results to stdout in a human readable format that is not suitable for verification (hence it is probably fine to keep it for stdout)
* timing (-t, --timing): print the timing of the execution: in the gpu version there are 3 times (copy to gpu, kernel execution, copy back to cpu), for now we only have one
* input (-i, --input): pass input data (using the hex format)
* help (-h, --help): print help information, taken care of by clap
* print-input (-p, --print-input): print the input matrix (or matrices) in a human readable format
* common but not implemented:
  - NOT CONSIDERED -g: not considered at the moment
  - NOT CONSIDERED -c, -C: having only one time, this does not seem useful
  - NOT RELEVANT -d: not relevant for cpu benchmarks

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

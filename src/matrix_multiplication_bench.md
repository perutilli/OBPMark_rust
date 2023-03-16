# MATRIX MULTIPLICATION BENCHMARK
## Questions
* Is it intended that we can only use square matrices? Or should I allow for different shapes?
* Why the hexadecimal representation in the input files? Should this remain the same or can I use some standard serialization format? Also not having any example file for this case makes it hard to verify that my implementation is correct. This function also exists `get_values_file(input_file, A, B)` which might be closer to what I was thinking of doing.
* Before thinking about any kind of serialization I would like to finalize the data structure that will hold the matrices. Can I use an "arbitrary" format or should I keep the hexadecimal representation that is used in the C version?
## Arguments
### Required
* `-s` (we should allow `--size` as well) - side size of the matrices supposing that they are square
### Optional
* `-v` - verification (compare the result with the naive sequential implementation, in our case I think we will use the sequential implementation in rust rather than the c one)
* `-e` - verification plus export results (TODO: to a file?)
* `-o` - print the output
* `-t` - print the timing
* `-c` - `csv_format` = true (TODO: check what it means)
* `-C` - `csv_format_timestamp` = true (TODO: check what it means)
* `-g` - `export_results_gpu` = true (TODO: check what it means)
* `-d` - number of gpus used ??
* `-f` - mute messages
* `-i` - input file names for A and B matrices
## Main body
1. Read the arguments
2. Initialize the matrices
   * If the input file names are not provided, generate random matrices. If `INT` is defined, use integers, otherwise use floats. If they are integers they are also modulo `NUMBER_BASE` * 100 (TODO: check this)
   * If the input file names are provided, read the matrices from the files using function `get_double_hexadecimal_values`
3. Stuff related to device (we don't care about this)
4. Run the function and time it. TODO: check how the various parameters should influence the behavior
5. Clean the memory (not necessary in rust)
## Input (and output) files format
The input files are in hexadecimal format. Each line contains an hexadecimal number. The output files are in the same format.
To read the matrix from the file, use function `get_double_hexadecimal_values`, that gets two bytes (characters 0-9,A-F) at a time into a temp vector and converts them to an integer value, which will be a byte in the number we are reading. This should be easy to do in rust, thanks to  `f32::from_bytes` and `f64::from_bytes`. However a consideration about endianess and type (i32, f32 or f64) should be made.
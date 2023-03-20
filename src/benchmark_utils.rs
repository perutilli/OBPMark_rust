use clap::Parser;

#[derive(Parser, Debug)]
pub struct CommonArgs {
    /// Size of the matrix (or matrices)
    #[arg(short, long)]
    pub size: usize,

    /// Export the result in hex format to file <export>
    #[arg(short, long)]
    pub export: Option<String>,

    /// Verifies the result against 2d reference implementation
    #[arg(short, long, default_value_t = false)]
    pub verification: bool,

    /// Prints the output in human readable format to stdout
    #[arg(short, long, default_value_t = false)]
    pub output: bool,

    /// Prints the kernel execution time to stdout
    #[arg(short, long, default_value_t = false)]
    pub timing: bool,

    /// Uses ["mat_A.in", ["mat_B.in"], "correct_C.in"] for input data and result verification
    #[arg(short, long)]
    pub input: Option<Vec<String>>,
}

use crate::Number;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

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

/// Parses the content of file at path and returns a 1d vector of Numbers
pub fn parse_input_file(path: &Path) -> Result<Vec<Number>, std::io::Error> {
    let file = File::open(path)?;
    let lines = io::BufReader::new(file).lines();
    let mut matrix = Vec::new();
    for line in lines {
        if let Ok(line) = line {
            // each line should contain 2 * size bytes (each byte is 2 hex digits)
            assert_eq!(line.len(), std::mem::size_of::<Number>() * 2);
            let values: Vec<_> = line
                .chars()
                .map(|c| c.to_digit(16).unwrap() as u8)
                .collect();
            let bytes: Vec<_> = values.chunks(2).map(|c| c[0] << 4 | c[1]).collect();
            // supposing that the input is in big endian (which I believe is what was used in the original implementation)
            matrix.push(Number::from_be_bytes(bytes.try_into().unwrap()));
        }
    }
    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_input_file() {
        let path = Path::new("src/test.in");
        let matrix = parse_input_file(path).unwrap();
        println!("{:?}", matrix);
        // Looks like it is working
        // Expected matrix:
        /*
        3.159429 3.283478 3.929763 3.173610 4.028711 3.145139 2.578660 3.173845 4.001301 4.187828
        2.971452 2.782580 3.744442 3.452793 3.564502 3.087291 2.929662 3.122181 3.978263 3.299551
        2.703999 2.758311 3.139623 3.326980 3.555843 3.160808 3.079007 2.839732 3.757029 3.009198
        1.801971 1.946579 2.537668 2.632910 2.607783 2.030808 2.320400 2.377404 2.620659 2.370991
        1.602856 2.104803 2.997047 2.959427 2.316515 2.543622 2.264620 1.983664 2.648273 3.097270
        1.894450 1.459126 2.546041 2.680983 2.262569 1.614490 2.334064 2.876295 3.108596 2.124824
        2.388071 1.958284 3.202265 2.546371 2.902442 1.713888 1.935139 3.105689 3.308853 2.977277
        1.789986 2.630796 2.699428 2.554984 2.787069 2.759730 2.206544 1.679993 2.468369 2.525814
        2.712769 1.920780 3.050027 2.642885 3.430038 2.698867 2.166742 2.872192 3.648272 2.706099
        2.207535 2.160978 2.820202 2.029182 2.592266 2.016126 2.026393 2.287798 2.853053 2.069752
        */
    }
}

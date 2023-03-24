use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

#[cfg(feature = "float")]
pub type Number = f32;
#[cfg(feature = "double")]
pub type Number = f64;
#[cfg(feature = "int")]
pub type Number = i32;
#[cfg(not(any(feature = "float", feature = "double", feature = "int")))]
pub type Number = f32;

#[derive(Debug)]
pub enum Error {
    InvalidDimensions,
}

#[derive(Debug)]
pub enum FileError {
    IoError(std::io::Error),
    InvalidSize(usize, usize),
    InvalidDatatype,
}

impl From<std::io::Error> for FileError {
    fn from(error: std::io::Error) -> Self {
        FileError::IoError(error)
    }
}

/// Trait that all matrix structs should implement
/// It contains basic methods such as `new`, `get_data`, `zeroes`
pub trait BaseMatrix {
    fn new(data: Vec<Vec<Number>>, rows: usize, cols: usize) -> Self;
    fn get_data(&self) -> Vec<Vec<Number>>;
    fn zeroes(rows: usize, cols: usize) -> Self
    where
        Self: Sized,
    {
        Self::new(vec![vec![Number::default(); cols]; rows], rows, cols)
    }

    fn from_random_seed(seed: u64, rows: usize, cols: usize) -> Self
    where
        Self: Sized,
    {
        let data = random_matrix_data(seed, rows, cols);
        Self::new(data, rows, cols)
    }

    fn from_file(path: &Path, rows: usize, cols: usize) -> Result<Self, FileError>
    where
        Self: Sized,
    {
        let file = File::open(path)?;
        let mut lines = io::BufReader::new(file).lines();
        let mut data = Vec::new();
        for i in 0..rows {
            let mut row = Vec::new();
            for j in 0..cols {
                let line = match lines.next() {
                    Some(line) => line?,
                    None => return Err(FileError::InvalidSize(i, j)),
                };
                if line.len() != std::mem::size_of::<Number>() * 2 {
                    return Err(FileError::InvalidDatatype);
                }
                let values: Vec<_> = line
                    .chars()
                    .map(|c| c.to_digit(16).unwrap() as u8)
                    .collect();
                let bytes: Vec<_> = values.chunks(2).map(|c| c[0] << 4 | c[1]).collect();
                // supposing that the input is in big endian (which I believe is what was used in the original implementation)
                row.push(Number::from_be_bytes(bytes.try_into().unwrap()));
            }
            data.push(row);
        }
        Ok(Self::new(data, rows, cols))
    }

    fn to_file(&self, path: &Path) -> Result<(), std::io::Error> {
        let mut file = File::create(path)?;
        for row in self.get_data() {
            for col in row {
                let bytes = col.to_be_bytes();
                let values = bytes
                    .iter()
                    .map(|b| format!("{:02x}", b))
                    .collect::<Vec<String>>()
                    .join("");
                writeln!(file, "{}", values)?;
            }
        }
        Ok(())
    }
}

/*
TODO: this is not possible like so, look here https://www.reddit.com/r/rust/comments/7qqbyp/comment/dss5b6z/
impl Display for BaseMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in self.get_data() {
            for el in row {
                // NOTE: this way we have a space before the newline, might not be what we want
                write!(f, "{} ", format_number(&el))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
*/

pub trait MatMul {
    fn multiply(&self, other: &Self, result: &mut Self) -> Result<(), Error>;
}

pub trait Relu {
    fn relu(&self, result: &mut Self) -> Result<(), Error>;
}

pub trait Softmax {
    fn softmax(&self, result: &mut Self) -> Result<(), Error>;
}

pub trait MaxPooling {
    fn max_pooling(
        &self,
        result: &mut Self,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<(), Error>;
}

pub fn random_matrix_data(seed: u64, rows: usize, cols: usize) -> Vec<Vec<Number>> {
    // NOTE: the seeding works only on the same machine/configuration
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![vec![Number::default(); cols]; rows];
    for row in &mut data {
        for col in row {
            // TODO: this should be constants/parameters
            let low: Number = "-10".parse().unwrap();
            let high: Number = "10".parse().unwrap();
            *col = rng.gen_range(low..high);
        }
    }
    data
}

#[cfg(feature = "int")]
pub fn format_number(number: &Number) -> String {
    // Ok for now, could make it a little more pretty
    // println!("{:10}: {}, {:?}", number, number, number.to_be_bytes());
    format!("{:5}", number)
}
#[cfg(not(feature = "int"))]
pub fn format_number(number: &Number) -> String {
    // Ok for now, could make it a little more pretty
    // println!("{:10}: {}, {:?}", number, number, number.to_be_bytes());
    format!("{:10.5}", number)
}

pub mod matrix_1d;
pub mod matrix_2d;
pub mod matrix_ndarray;

pub mod benchmark_utils;

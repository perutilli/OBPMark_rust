use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

use number_traits::*;

#[derive(Debug)]
pub enum Error {
    InvalidDimensions,
    InvalidKernelDimensions,
    InvalidNumberOfThreads,
}

#[derive(Debug)]
pub enum FileError {
    IoError(std::io::Error),
    InvalidSize(usize, usize),
    InvalidDatatype,
}

pub enum Padding {
    Zeroes,
}

impl From<std::io::Error> for FileError {
    fn from(error: std::io::Error) -> Self {
        FileError::IoError(error)
    }
}

/// Trait that all matrix structs should implement
/// It contains basic methods such as `new`, `get_data`, `zeroes`
pub trait BaseMatrix<T: Number> {
    fn new(data: Vec<Vec<T>>, rows: usize, cols: usize) -> Self;
    fn get_data(&self) -> Vec<Vec<T>>;
    fn zeroes(rows: usize, cols: usize) -> Self
    where
        Self: Sized,
    {
        Self::new(vec![vec![T::zero(); cols]; rows], rows, cols)
    }

    fn from_random_seed(seed: u64, rows: usize, cols: usize, min: T, max: T) -> Self
    where
        Self: Sized,
    {
        let data = random_matrix_data(seed, rows, cols, min, max);
        Self::new(data, rows, cols)
    }

    fn from_file(path: &Path, rows: usize, cols: usize) -> Result<Self, FileError>
    where
        Self: Sized,
        <T as Serialize>::Bytes: TryFrom<Vec<u8>>,
        <<T as Serialize>::Bytes as TryFrom<Vec<u8>>>::Error: std::fmt::Debug,
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
                if line.len() != std::mem::size_of::<T>() * 2 {
                    return Err(FileError::InvalidDatatype);
                }
                let values: Vec<_> = line
                    .chars()
                    .map(|c| c.to_digit(16).unwrap() as u8)
                    .collect();
                let bytes: Vec<_> = values.chunks(2).map(|c| c[0] << 4 | c[1]).collect();
                // supposing that the input is in big endian (which I believe is what was used in the original implementation)
                row.push(T::from_be_bytes(bytes.try_into().unwrap()));
            }
            data.push(row);
        }
        Ok(Self::new(data, rows, cols))
    }

    fn to_file(&self, path: &Path) -> Result<(), std::io::Error>
    where
        <T as Serialize>::Bytes: IntoIterator<Item = u8>,
    {
        let mut file = File::create(path)?;
        for row in self.get_data() {
            for col in row {
                let bytes = col.to_be_bytes();
                let values = bytes
                    .into_iter()
                    .map(|b| format!("{:02x}", b))
                    .collect::<Vec<String>>()
                    .join("");
                writeln!(file, "{}", values)?;
            }
        }
        Ok(())
    }

    fn reshape(&mut self, new_rows: usize, new_cols: usize) -> Result<(), Error>;
}

macro_rules! impl_display {
    ($t:ident) => {
        impl<T: Number> std::fmt::Display for $t<T> {
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
    };
    () => {};
}

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

// TODO: check with Leonidas the expected precision of this
// in the C version it is Num: i32 -> f32, Num: f32 -> f32, Num: f64 -> f64
// for semplicity I am going to use f64 for all for now
pub trait Correlation {
    fn correlation(&self, other: &Self) -> Result<f64, Error>;
}

pub trait Convolution {
    fn convolute(&self, kernel: &Self, padding: Padding, result: &mut Self) -> Result<(), Error>;
}

pub trait LRN<T: Float> {
    fn lrn(&self, result: &mut Self, alpha: T, beta: T, k: T) -> Result<(), Error>;
}

pub trait FastFourierTransform {
    fn fft(&mut self, nn: usize, start_pos: usize) -> Result<(), Error>;
}

pub trait FastFourierTransformWindowed: FastFourierTransform {
    fn fftw(&mut self, nn: usize, window: usize, result: &mut Self) -> Result<(), Error>;
}

pub trait WaveletTransformInteger<T: Integer> {
    fn wavelet_transform(&self, result: &mut Self, size: usize) -> Result<(), Error>;
}

pub trait WaveletTransformFloating<T: Float> {
    fn wavelet_transform(
        &self,
        result: &mut Self,
        size: usize,
        low_pass_filter: &[T],
        low_pass_filter_size: usize,
        high_pass_filter: &[T],
        high_pass_filter_size: usize,
    ) -> Result<(), Error>;
}

pub trait ParallelMatMul {
    fn parallel_multiply(
        &self,
        other: &Self,
        result: &mut Self,
        n_threads: usize,
    ) -> Result<(), Error>;
}

pub fn random_matrix_data<T: Number>(
    seed: u64,
    rows: usize,
    cols: usize,
    min: T,
    max: T,
) -> Vec<Vec<T>> {
    // NOTE: the seeding works only on the same machine/configuration
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![vec![T::zero(); cols]; rows];
    for row in &mut data {
        for col in row {
            *col = T::gen_range(&mut rng, min, max);
        }
    }
    data
}

// TODO: cfg should not be in the lib part, we should configure this some other way
#[cfg(feature = "int")]
pub fn format_number<T: Number>(number: &T) -> String {
    format!("{:5}", number)
}
#[cfg(not(feature = "int"))]
pub fn format_number<T: Number>(number: &T) -> String {
    format!("{:10.5}", number)
}

pub mod rayon_traits;

pub mod matrix_1d;
pub mod matrix_2d;

pub mod benchmark_utils;
pub mod number_traits;

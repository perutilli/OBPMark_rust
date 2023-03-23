use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
}

pub trait MatMul {
    fn multiply(&self, other: &Self, result: &mut Self) -> Result<(), Error>;
}

pub trait Relu {
    fn relu(&self, result: &mut Self) -> Result<(), Error>;
}

pub fn random_matrix(seed: u64, rows: usize, cols: usize) -> Vec<Vec<Number>> {
    // NOTE: the seeding works only on the same machine/configuration
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![vec![Number::default(); cols]; rows];
    for row in &mut data {
        for col in row {
            // TODO: this should be constants/parameters
            let low: Number = "-100".parse().unwrap();
            let high: Number = "100".parse().unwrap();
            *col = rng.gen_range(low..high);
        }
    }
    data
}

pub fn format_number(number: &Number) -> String {
    // Ok for now, could make it a little more pretty
    format!("{:<10}", number)
}

// TODO: we might want to change this, ok for now
pub trait FromRandomSeed {
    fn from_random_seed(seed: u64, rows: usize, cols: usize) -> Self;
}

pub mod matrix_1d;
pub mod matrix_2d;
pub mod matrix_ndarray;

pub mod benchmark_utils;

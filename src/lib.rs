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

// TODO: we might want to change this, ok for now
// TODO: right now we will overflow for ints, need to fix that
pub trait FromRandomSeed {
    fn from_random_seed(seed: u64, rows: usize, cols: usize) -> Self;
}

pub mod matrix_1d;
pub mod matrix_2d;
pub mod matrix_ndarray;

pub mod benchmark_utils;

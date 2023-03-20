use std::fmt::Display;

use crate::{random_matrix, Error, FromRandomSeed, MatMul, Number};

pub struct Matrix {
    data: Vec<Vec<Number>>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<Vec<Number>>, rows: usize, cols: usize) -> Matrix {
        Matrix { data, rows, cols }
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in &self.data {
            for col in row {
                // NOTE: this way we have a space before the newline, might not be what we want
                write!(f, "{} ", col)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl MatMul for Matrix {
    fn multiply(&self, other: &Matrix, result: &mut Matrix) -> Result<(), Error> {
        if self.cols != other.rows {
            return Err(Error::InvalidDimensions);
        }

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = Number::default();
                // NOTE: this allows result to not be all zeros
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        Ok(())
    }
}

impl FromRandomSeed for Matrix {
    fn from_random_seed(seed: u64, rows: usize, cols: usize) -> Matrix {
        let data = random_matrix(seed, rows, cols);
        Matrix::new(data, rows, cols)
    }
}

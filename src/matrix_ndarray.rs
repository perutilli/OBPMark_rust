use std::fmt::Display;

use ndarray::Array2;

use crate::{random_matrix, Error, FromRandomSeed, MatMul, Number};

pub struct Matrix {
    data: Array2<Number>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<Number>, rows: usize, cols: usize) -> Matrix {
        Matrix {
            data: Array2::from_shape_vec((rows, cols), data).unwrap(),
            rows,
            cols,
        }
    }

    pub fn get_data(&self) -> Vec<Vec<Number>> {
        self.data.outer_iter().map(|x| x.to_vec()).collect()
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in self.data.outer_iter() {
            for el in row {
                // NOTE: this way we have a space before the newline, might not be what we want
                write!(f, "{:.6} ", el)?;
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

        result.data = self.data.dot(&other.data);
        Ok(())
    }
}

impl FromRandomSeed for Matrix {
    fn from_random_seed(seed: u64, rows: usize, cols: usize) -> Matrix {
        let data = random_matrix(seed, rows, cols);
        Matrix::new(data.into_iter().flatten().collect(), rows, cols)
    }
}

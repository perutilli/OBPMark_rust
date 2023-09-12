use crate::{BaseMatrix, Error, Number};

pub struct Matrix2d<T: Number> {
    data: Vec<Vec<T>>,
    rows: usize,
    cols: usize,
}

impl<T: Number> BaseMatrix<T> for Matrix2d<T> {
    fn new(data: Vec<Vec<T>>, rows: usize, cols: usize) -> Matrix2d<T> {
        Matrix2d { data, rows, cols }
    }

    fn get_data(&self) -> Vec<Vec<T>> {
        self.data.clone()
    }

    fn reshape(&mut self, new_rows: usize, new_cols: usize) -> Result<(), Error> {
        println!("WARNING: reshape is an expensive operation for 2d matrices");
        if self.rows * self.cols != new_rows * new_cols {
            return Err(Error::InvalidDimensions);
        }
        let old_data = self.data.clone();
        let flat_data: Vec<_> = old_data.into_iter().flatten().collect();
        self.data = flat_data
            .chunks(new_cols)
            .map(|chunk| chunk.to_vec())
            .collect();
        self.rows = new_rows;
        self.cols = new_cols;
        Ok(())
    }

    fn transpose(&self) -> Self {
        let mut data = vec![vec![T::zero(); self.rows]; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j][i] = self.data[i][j];
            }
        }
        Matrix2d {
            data,
            rows: self.cols,
            cols: self.rows,
        }
    }
}

impl_display!(Matrix2d);

mod parallel_implementations;
mod rayon_implementations;
mod sequential_implementations;

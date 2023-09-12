use crate::{BaseMatrix, Error, Number};

pub struct Matrix1d<T: Number> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Number> BaseMatrix<T> for Matrix1d<T> {
    fn new(data: Vec<Vec<T>>, rows: usize, cols: usize) -> Matrix1d<T> {
        Matrix1d {
            data: data.into_iter().flatten().collect(),
            rows,
            cols,
        }
    }

    fn get_data(&self) -> Vec<Vec<T>> {
        self.data
            .clone()
            .chunks(self.cols)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<T>>>()
    }

    fn reshape(&mut self, new_rows: usize, new_cols: usize) -> Result<(), Error> {
        if new_rows * new_cols != self.rows * self.cols {
            return Err(Error::InvalidDimensions);
        }
        self.rows = new_rows;
        self.cols = new_cols;
        Ok(())
    }

    fn transpose(&self) -> Self {
        let mut data = vec![T::zero(); self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix1d {
            data,
            rows: self.cols,
            cols: self.rows,
        }
    }
}

impl_display!(Matrix1d);

mod parallel_implementations;
mod rayon_implementations;
mod sequential_implementations;

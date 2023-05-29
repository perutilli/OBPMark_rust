extern crate nalgebra as na;

use crate::{BaseMatrix, Error, Number};
use na::{DMatrix, Dyn};

use crate::MatMul;

pub struct MatrixNalgebra<T: Number> {
    data: DMatrix<T>,
    rows: usize,
    cols: usize,
}

impl<T: Number> BaseMatrix<T> for MatrixNalgebra<T> {
    fn new(data: Vec<Vec<T>>, rows: usize, cols: usize) -> Self {
        let data = DMatrix::from_row_slice(
            rows,
            cols,
            &data.into_iter().flatten().collect::<Vec<_>>()[..],
        );
        MatrixNalgebra { data, rows, cols }
    }

    fn get_data(&self) -> Vec<Vec<T>> {
        self.data.clone().iter().map(|x| vec![*x]).collect()
    }

    fn reshape(&mut self, new_rows: usize, new_cols: usize) -> Result<(), Error> {
        if new_rows * new_cols != self.rows * self.cols {
            return Err(Error::InvalidDimensions);
        }

        let mut data: DMatrix<T> = DMatrix::zeros(1, 1);
        std::mem::swap(&mut self.data, &mut data);
        self.data = data.reshape_generic(Dyn(new_rows), Dyn(new_cols));
        Ok(())
    }
}

impl_display!(MatrixNalgebra);

impl<T: Number> MatMul for MatrixNalgebra<T> {
    fn multiply(&self, other: &Self, result: &mut Self) -> Result<(), Error> {
        // TODO: check dimensions
        result.data = &self.data * &other.data;
        Ok(())
    }
}

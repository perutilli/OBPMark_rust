use std::fmt::Display;

use ndarray::Array2;

use crate::{format_number, BaseMatrix, Num};

pub struct MatrixNdArray<T: Num> {
    data: Array2<T>,
    /*
    rows: usize,
    cols: usize,
     */
}

impl<T: Num> BaseMatrix<T> for MatrixNdArray<T> {
    fn new(data: Vec<Vec<T>>, rows: usize, cols: usize) -> Self {
        MatrixNdArray {
            data: Array2::from_shape_vec((rows, cols), data.into_iter().flatten().collect())
                .unwrap(),
            // rows,
            // cols,
        }
    }

    fn get_data(&self) -> Vec<Vec<T>> {
        self.data.outer_iter().map(|x| x.to_vec()).collect()
    }
}

impl<T: Num> Display for MatrixNdArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in self.data.outer_iter() {
            for el in row {
                // NOTE: this way we have a space before the newline, might not be what we want
                write!(f, "{} ", format_number(el))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/*
impl<T: Num> MatMul for Matrix<T> {
    fn multiply(&self, other: &Matrix<T>, result: &mut Matrix<T>) -> Result<(), Error> {
        if self.cols != other.rows {
            return Err(Error::InvalidDimensions);
        }

        result.data = self.data.dot(&other.data);
        Ok(())
    }
}
 */

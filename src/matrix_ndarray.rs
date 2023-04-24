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

    fn set(&mut self, row: usize, col: usize, value: T) {
        unimplemented!("{} {} {}", row, col, value);
    }
}

impl_display!(MatrixNdArray);

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

use std::fmt::Display;

use num::Float;

use crate::{format_number, BaseMatrix, Error, MatMul, MaxPooling, Num, Relu, Softmax};

pub struct Matrix<T: Num> {
    data: Vec<Vec<T>>,
    rows: usize,
    cols: usize,
}

impl<T: Num> BaseMatrix<T> for Matrix<T> {
    fn new(data: Vec<Vec<T>>, rows: usize, cols: usize) -> Matrix<T> {
        Matrix { data, rows, cols }
    }

    fn get_data(&self) -> Vec<Vec<T>> {
        self.data.clone()
    }
}

impl<T: Num> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in &self.data {
            for el in row {
                // NOTE: this way we have a space before the newline, might not be what we want
                write!(f, "{} ", format_number(el))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl<T: Num> MatMul for Matrix<T> {
    fn multiply(&self, other: &Matrix<T>, result: &mut Matrix<T>) -> Result<(), Error> {
        if self.cols != other.rows {
            return Err(Error::InvalidDimensions);
        }

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::default();
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

impl<T: Num> Relu for Matrix<T> {
    fn relu(&self, result: &mut Matrix<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i][j] < T::zero() {
                    result.data[i][j] = T::zero();
                } else {
                    result.data[i][j] = self.data[i][j];
                }
                // result.data[i][j] = self.data[i][j].max(T::default());
            }
        }
        Ok(())
    }
}

#[cfg(not(feature = "int"))]
impl<T: Num + Float> Softmax for Matrix<T> {
    fn softmax(&self, result: &mut Matrix<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            let mut sum = T::default();
            for j in 0..self.cols {
                let val = self.data[i][j].exp();
                sum += val;
                result.data[i][j] = val;
            }
            for j in 0..self.cols {
                result.data[i][j] /= sum;
            }
        }
        Ok(())
    }
}

impl<T: Num> MaxPooling for Matrix<T> {
    fn max_pooling(
        &self,
        result: &mut Matrix<T>,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<(), Error> {
        if self.rows != result.rows * row_stride || self.cols != result.cols * col_stride {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..result.rows {
            for j in 0..result.cols {
                let mut max = self.data[i * row_stride][j * col_stride];
                for k in 0..row_stride {
                    for l in 0..col_stride {
                        if max < self.data[i * row_stride + k][j * col_stride + l] {
                            max = self.data[i * row_stride + k][j * col_stride + l];
                        }
                        // max = max.max(self.data[i * row_stride + k][j * col_stride + l]);
                    }
                }
                result.data[i][j] = max;
            }
        }
        Ok(())
    }
}

use std::fmt::Display;

use crate::{format_number, BaseMatrix, Error, MatMul, MaxPooling, Number, Relu, Softmax};

pub struct Matrix {
    data: Vec<Vec<Number>>,
    rows: usize,
    cols: usize,
}

impl BaseMatrix for Matrix {
    fn new(data: Vec<Vec<Number>>, rows: usize, cols: usize) -> Matrix {
        Matrix { data, rows, cols }
    }

    fn get_data(&self) -> Vec<Vec<Number>> {
        self.data.clone()
    }
}

impl Display for Matrix {
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

impl Relu for Matrix {
    fn relu(&self, result: &mut Matrix) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j].max(Number::default());
            }
        }
        Ok(())
    }
}

#[cfg(not(feature = "int"))]
impl Softmax for Matrix {
    fn softmax(&self, result: &mut Matrix) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            let mut sum = Number::default();
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

impl MaxPooling for Matrix {
    fn max_pooling(
        &self,
        result: &mut Matrix,
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
                        max = max.max(self.data[i * row_stride + k][j * col_stride + l]);
                    }
                }
                result.data[i][j] = max;
            }
        }
        Ok(())
    }
}

use num::Float;

use crate::{
    format_number, BaseMatrix, Correlation, Error, MatMul, MaxPooling, Num, Relu, Softmax,
};

pub struct Matrix2d<T: Num> {
    data: Vec<Vec<T>>,
    rows: usize,
    cols: usize,
}

impl<T: Num> BaseMatrix<T> for Matrix2d<T> {
    fn new(data: Vec<Vec<T>>, rows: usize, cols: usize) -> Matrix2d<T> {
        Matrix2d { data, rows, cols }
    }

    fn get_data(&self) -> Vec<Vec<T>> {
        self.data.clone()
    }
}

impl_display!(Matrix2d);

impl<T: Num> MatMul for Matrix2d<T> {
    fn multiply(&self, other: &Matrix2d<T>, result: &mut Matrix2d<T>) -> Result<(), Error> {
        if self.cols != other.rows {
            return Err(Error::InvalidDimensions);
        }

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::zero();
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

impl<T: Num> Relu for Matrix2d<T> {
    fn relu(&self, result: &mut Matrix2d<T>) -> Result<(), Error> {
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
impl<T: Num + Float> Softmax for Matrix2d<T> {
    fn softmax(&self, result: &mut Matrix2d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            let mut sum = T::zero();
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

impl<T: Num> MaxPooling for Matrix2d<T> {
    fn max_pooling(
        &self,
        result: &mut Matrix2d<T>,
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

impl<T: Num> Correlation for Matrix2d<T> {
    fn correlation(&self, other: &Matrix2d<T>) -> Result<f64, Error> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(Error::InvalidDimensions);
        }

        let mut acc_self_sq = 0_f64;
        let mut acc_other_sq = 0_f64;
        let mut acc_self_other = 0_f64;

        let self_mean =
            self.data.iter().flatten().sum::<T>().as_f64() / (self.rows * self.cols) as f64;
        let other_mean =
            other.data.iter().flatten().sum::<T>().as_f64() / (other.rows * other.cols) as f64;

        for i in 0..self.rows {
            for j in 0..self.cols {
                let self_delta = self.data[i][j].as_f64() - self_mean;
                let other_delta = other.data[i][j].as_f64() - other_mean;
                acc_self_sq += self_delta * self_delta;
                acc_other_sq += other_delta * other_delta;
                acc_self_other += self_delta * other_delta;
            }
        }
        Ok(acc_self_other / (acc_self_sq * acc_other_sq).sqrt())
    }
}

use num::Float;

use crate::{format_number, BaseMatrix, Error, MatMul, MaxPooling, Num, Relu, Softmax};

pub struct Matrix1d<T: Num> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Num> BaseMatrix<T> for Matrix1d<T> {
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
}

impl_display!(Matrix1d);

impl<T: Num> MatMul for Matrix1d<T> {
    fn multiply(&self, other: &Matrix1d<T>, result: &mut Matrix1d<T>) -> Result<(), Error> {
        if self.cols != other.rows {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::zero();
                // NOTE: this allows result to not be all zeros
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * self.rows + j] = sum;
            }
        }
        Ok(())
    }
}

impl<T: Num> Relu for Matrix1d<T> {
    fn relu(&self, result: &mut Matrix1d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i * self.rows + j] > T::zero() {
                    result.data[i * self.rows + j] = self.data[i * self.rows + j];
                } else {
                    result.data[i * self.rows + j] = T::zero();
                }
                // result.data[i * self.rows + j] = self.data[i * self.rows + j].max(T::default());
            }
        }
        Ok(())
    }
}

#[cfg(not(feature = "int"))]
// TODO: check that Float is what we want
impl<T: Num + Float> Softmax for Matrix1d<T> {
    fn softmax(&self, result: &mut Matrix1d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            let mut sum = T::zero();
            for j in 0..self.cols {
                let val = self.data[i * self.rows + j].exp();
                sum += val;
                result.data[i * self.rows + j] = val;
            }
            for j in 0..self.cols {
                result.data[i * self.rows + j] /= sum;
            }
        }
        Ok(())
    }
}

impl<T: Num> MaxPooling for Matrix1d<T> {
    fn max_pooling(
        &self,
        result: &mut Matrix1d<T>,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<(), Error> {
        if self.rows != result.rows * row_stride || self.cols != result.cols * col_stride {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..result.rows {
            for j in 0..result.cols {
                let mut max = self.data[i * row_stride * self.cols + j * col_stride];
                for k in 0..row_stride {
                    for l in 0..col_stride {
                        if max
                            < self.data
                                [i * row_stride * self.cols + j * col_stride + k * self.cols + l]
                        {
                            max = self.data
                                [i * row_stride * self.cols + j * col_stride + k * self.cols + l];
                        }
                        /*
                        max = max.max(
                            self.data
                                [i * row_stride * self.cols + j * col_stride + k * self.cols + l],
                        );
                         */
                    }
                }
                result.data[i * result.cols + j] = max;
            }
        }
        Ok(())
    }
}

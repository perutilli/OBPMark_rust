use num::Float;

use crate::{
    format_number, BaseMatrix, Convolution, Correlation, Error, MatMul, MaxPooling, Num, Relu,
    Softmax,
};

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

impl<T: Num> Correlation for Matrix1d<T> {
    fn correlation(&self, other: &Matrix1d<T>) -> Result<f64, Error> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(Error::InvalidDimensions);
        }

        let mut acc_self_sq = 0_f64;
        let mut acc_other_sq = 0_f64;
        let mut acc_self_other = 0_f64;

        let self_mean = self.data.iter().sum::<T>().as_f64() / (self.rows * self.cols) as f64;
        let other_mean = other.data.iter().sum::<T>().as_f64() / (other.rows * other.cols) as f64;

        for i in 0..self.rows {
            for j in 0..self.cols {
                let self_delta = self.data[i * self.cols + j].as_f64() - self_mean;
                let other_delta = other.data[i * self.cols + j].as_f64() - other_mean;
                acc_self_sq += self_delta * self_delta;
                acc_other_sq += other_delta * other_delta;
                acc_self_other += self_delta * other_delta;
            }
        }
        Ok(acc_self_other / (acc_self_sq * acc_other_sq).sqrt())
    }
}

use crate::Padding;
impl<T: Num> Convolution for Matrix1d<T> {
    fn convolute(&self, kernel: &Self, padding: Padding, result: &mut Self) -> Result<(), Error> {
        match padding {
            Padding::Zeroes => (),
        }

        if self.rows != result.rows || self.cols != result.cols {
            // NOTE: this is a very specific kind of convolution, we probably want to support
            //       more general cases, in particular at least the no padding case
            return Err(Error::InvalidDimensions);
        }

        if kernel.rows % 2 == 0 || kernel.cols % 2 == 0 {
            return Err(Error::InvalidKernelDimensions);
        }

        let kernel_y_radius = (kernel.rows - 1) / 2;
        let kernel_x_radius = (kernel.cols - 1) / 2;

        for i in 0..self.rows {
            for j in 0..self.cols {
                let mut sum = T::zero();
                for k in 0..kernel.rows {
                    for l in 0..kernel.cols {
                        let y = (i + k) as isize - kernel_y_radius as isize;
                        let x = (j + l) as isize - kernel_x_radius as isize;
                        if (y > 0 && y < self.rows as isize) && (x > 0 && x < self.cols as isize) {
                            let y = y as usize;
                            let x = x as usize;
                            sum += self.data[y * self.cols + x] * kernel.data[k * kernel.cols + l];
                        }
                    }
                }
                result.data[i * result.cols + j] = sum;
            }
        }
        Ok(())
    }
}

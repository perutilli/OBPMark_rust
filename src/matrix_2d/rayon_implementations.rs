use crate::matrix_2d::Matrix2d;
use crate::number_traits::{Float, Number};
use crate::rayon_traits::*;
use crate::Error;

use rayon::prelude::*;

impl<T: Number> RayonMatMul for Matrix2d<T> {
    fn rayon_multiply(&self, other: &Self, result: &mut Self) -> Result<(), Error> {
        if self.cols != other.rows || self.rows != result.rows || other.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result.data.par_iter_mut().enumerate().for_each(|(i, row)| {
            for (j, el) in row.iter_mut().enumerate() {
                let mut sum = T::zero();
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                *el = sum;
            }
        });

        Ok(())
    }
}

impl<T: Number> RayonMaxPooling for Matrix2d<T> {
    fn rayon_max_pooling(
        &self,
        result: &mut Matrix2d<T>,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<(), Error> {
        if self.rows != result.rows * row_stride || self.cols != result.cols * col_stride {
            return Err(Error::InvalidDimensions);
        }
        // for i in 0..result.rows {
        result.data.par_iter_mut().enumerate().for_each(|(i, row)| {
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
                row[j] = max;
            }
        });
        Ok(())
    }
}

impl<T: Float> RayonSoftmax for Matrix2d<T> {
    fn rayon_softmax(&self, result: &mut Matrix2d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        // for i in 0..self.rows {
        result.data.par_iter_mut().enumerate().for_each(|(i, row)| {
            let mut sum = T::zero();
            for j in 0..self.cols {
                let val = self.data[i][j].exp();
                sum += val;
                row[j] = val;
            }
            for j in 0..self.cols {
                row[j] /= sum;
            }
        });
        Ok(())
    }
}

use crate::Padding;
impl<T: Number> RayonConvolution for Matrix2d<T> {
    fn rayon_convolute(
        &self,
        kernel: &Self,
        padding: Padding,
        result: &mut Self,
    ) -> Result<(), Error> {
        match padding {
            Padding::Zeroes => (),
        }

        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }

        if kernel.rows % 2 == 0 || kernel.cols % 2 == 0 {
            return Err(Error::InvalidKernelDimensions);
        }

        let kernel_y_radius = (kernel.rows - 1) / 2;
        let kernel_x_radius = (kernel.cols - 1) / 2;

        //for i in 0..self.rows {
        result.data.par_iter_mut().enumerate().for_each(|(i, row)| {
            for j in 0..self.cols {
                let mut sum = T::zero();
                for k in 0..kernel.rows {
                    for l in 0..kernel.cols {
                        let y = (i + k) as isize - kernel_y_radius as isize;
                        let x = (j + l) as isize - kernel_x_radius as isize;
                        if (y > 0 && y < self.rows as isize) && (x > 0 && x < self.cols as isize) {
                            let y = y as usize;
                            let x = x as usize;
                            sum += self.data[y][x] * kernel.data[k][l];
                        }
                    }
                }
                row[j] = sum;
            }
        });
        Ok(())
    }
}

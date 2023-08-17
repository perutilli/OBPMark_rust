use crate::matrix_2d::Matrix2d;
use crate::number_traits::{Float, Number};
use crate::Error;
use crate::{rayon_traits::*, FirFilter};

use rayon::prelude::*;

use crate::{Convolution, MatMul, MaxPooling, Relu, Softmax, LRN};

impl<T: Number> RayonMatMul for Matrix2d<T> {
    fn rayon_multiply(&self, other: &Self, result: &mut Self) -> Result<(), Error> {
        if self.cols != other.rows || self.rows != result.rows || other.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, row)| self.multiply_row(other, row, i));

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
            self.max_pooling_row(row, i, row_stride, col_stride);
        });
        Ok(())
    }
}

impl<T: Float> RayonSoftmax for Matrix2d<T> {
    fn rayon_softmax(&self, result: &mut Matrix2d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }

        let sum = result
            .data
            .par_iter_mut()
            .enumerate()
            .map(|(i, row)| {
                self.softmax_row(row, i) // SIDE EFFECT ON ROW
            })
            .reduce(|| T::zero(), |partial_sum, next_sum| partial_sum + next_sum);

        result.data.par_iter_mut().flatten().for_each(|el| {
            *el /= sum;
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

        result
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, row)| self.convolute_row(kernel, row, i));
        Ok(())
    }
}

impl<T: Number> RayonRelu for Matrix2d<T> {
    fn rayon_relu(&self, result: &mut Matrix2d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, row)| self.relu_row(row, i));
        Ok(())
    }
}

impl<T: Float> RayonLRN<T> for Matrix2d<T> {
    fn rayon_lrn(&self, result: &mut Matrix2d<T>, alpha: T, beta: T, k: T) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result.data.par_iter_mut().enumerate().for_each(|(i, row)| {
            self.lrn_row(row, i, alpha, beta, k);
        });
        Ok(())
    }
}

impl<T: Number> RayonFiniteImpulseResponseFilter for Matrix2d<T> {
    fn rayon_fir_filter(&self, kernel: &Self, result: &mut Self) -> Result<(), Error> {
        if self.rows != result.rows || result.cols != self.cols + kernel.cols - 1 || self.rows != 1
        {
            return Err(Error::InvalidDimensions);
        }
        if kernel.rows != 1 {
            return Err(Error::InvalidKernelDimensions);
        }

        result.data[0] // All the data is in data[0] because the matrix is one-dimensional
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, el)| {
                *el = self.fir_filter_element(kernel, idx);
            });
        Ok(())
    }
}

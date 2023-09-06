use crate::matrix_1d::Matrix1d;
use crate::number_traits::{Float, Number};
use crate::Error;
use crate::{rayon_traits::*, FirFilter};

use crate::{Convolution, FastFourierTransformHelper, MatMul, MaxPooling, Relu, Softmax, LRN};

use rayon::prelude::*;

impl<T: Number> RayonMatMul for Matrix1d<T> {
    fn rayon_multiply(&self, other: &Self, result: &mut Self) -> Result<(), Error> {
        if self.cols != other.rows || self.rows != result.rows || other.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .par_chunks_mut(other.cols)
            .enumerate()
            .for_each(|(i, chunk)| {
                self.multiply_row(other, chunk, i);
            });

        Ok(())
    }
}

impl<T: Number> RayonMaxPooling for Matrix1d<T> {
    fn rayon_max_pooling(
        &self,
        result: &mut Matrix1d<T>,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<(), Error> {
        if self.rows != result.rows * row_stride || self.cols != result.cols * col_stride {
            return Err(Error::InvalidDimensions);
        }
        // for i in 0..result.rows {
        result
            .data
            .par_chunks_mut(result.cols)
            .enumerate()
            .for_each(|(i, row)| self.max_pooling_row(row, i, row_stride, col_stride));

        Ok(())
    }
}

impl<T: Float> RayonSoftmax for Matrix1d<T> {
    fn rayon_softmax(&self, result: &mut Matrix1d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }

        let sum = result
            .data
            .par_chunks_mut(self.cols)
            .enumerate()
            .map(|(i, row)| self.softmax_row(row, i)) // note that this map operation has side effects (i.e. calculating the exp of each element), not super pretty
            .reduce(|| T::zero(), |partial_sum, next_sum| partial_sum + next_sum);

        // here we do need for the whole sum to be computed, so we need to wait before normalizing
        result.data.par_iter_mut().for_each(|el| {
            *el /= sum;
        });

        Ok(())
    }
}

use crate::Padding;
impl<T: Number> RayonConvolution for Matrix1d<T> {
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
            .par_chunks_mut(result.cols)
            .enumerate()
            .for_each(|(i, row)| {
                self.convolute_row(kernel, row, i);
            });
        Ok(())
    }
}

impl<T: Number> RayonRelu for Matrix1d<T> {
    fn rayon_relu(&self, result: &mut Matrix1d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .par_chunks_mut(self.cols)
            .enumerate()
            .for_each(|(i, row)| self.relu_row(row, i));
        Ok(())
    }
}

impl<T: Float> RayonLRN<T> for Matrix1d<T> {
    fn rayon_lrn(&self, result: &mut Self, alpha: T, beta: T, k: T) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .par_chunks_mut(self.cols)
            .enumerate()
            .for_each(|(i, row)| {
                self.lrn_row(row, i, alpha, beta, k);
            });
        Ok(())
    }
}

impl<T: Number> RayonFiniteImpulseResponseFilter for Matrix1d<T> {
    fn rayon_fir_filter(&self, kernel: &Self, result: &mut Self) -> Result<(), Error> {
        if result.cols != self.cols + kernel.cols - 1 || self.rows != 1 || result.rows != 1 {
            return Err(Error::InvalidDimensions);
        }
        if kernel.rows != 1 {
            return Err(Error::InvalidKernelDimensions);
        }

        result
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, el)| {
                *el = self.fir_filter_element(kernel, idx);
            });
        Ok(())
    }
}

macro_rules! impl_rayon_fft_windowed {
    ($t: tt) => {
        impl RayonFastFourierTransformWindowed<$t> for Matrix1d<$t> {
            fn rayon_fft_windowed(&self, window: usize, result: &mut Self) -> Result<(), Error> {
                if self.rows != 1 || result.rows != 1 {
                    return Err(Error::InvalidDimensions);
                }

                result
                    .data
                    .par_chunks_mut(window * 2)
                    .enumerate()
                    .for_each(|(i, result_chunk)| {
                        for j in 0..window {
                            result_chunk[j] = self.data[i * 2 + j];
                        }
                        Self::fft_helper(result_chunk, window >> 1);
                    });
                Ok(())
            }
        }
    };
}

impl_rayon_fft_windowed!(f32);
impl_rayon_fft_windowed!(f64);

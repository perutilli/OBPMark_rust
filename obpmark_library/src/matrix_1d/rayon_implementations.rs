use crate::matrix_1d::Matrix1d;
use crate::number_traits::{Float, Number};
use crate::Error;
use crate::{rayon_traits::*, FirFilter};

use crate::{
    BaseMatrix, Convolution, Correlation, FastFourierTransformHelper, MatMul, MaxPooling, Relu,
    Softmax, LRN,
};

use rayon::prelude::*;

impl<T: Number> RayonMatMul for Matrix1d<T> {
    fn rayon_multiply(&self, other: &Self, result: &mut Self) -> Result<(), Error> {
        if self.cols != other.rows || self.rows != result.rows || other.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }

        let other_transposed = other.transpose();

        result
            .data
            .par_chunks_mut(other.cols)
            .enumerate()
            .for_each(|(i, chunk)| {
                self.multiply_row(&other_transposed, chunk, i);
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

macro_rules! impl_rayon_corr {
    ($self_type: tt, $output_type: tt) => {
        impl RayonCorrelation for Matrix1d<$self_type> {
            type Output = $output_type;
            fn rayon_correlate(&self, other: &Self) -> Result<Self::Output, Error> {
                if self.rows != other.rows || self.cols != other.cols {
                    return Err(Error::InvalidDimensions);
                }

                let self_mean = self.data.par_iter().sum::<$self_type>() as Self::Output
                    / (self.rows * self.cols) as Self::Output;
                let other_mean = other.data.par_iter().sum::<$self_type>() as Self::Output
                    / (other.rows * other.cols) as Self::Output;

                let (acc_self_sq, acc_other_sq, acc_self_other) = (0..self.rows)
                    .into_par_iter()
                    .map(|i| self.accumulate_row(&other, self_mean, other_mean, i))
                    .reduce(
                        || (0.0, 0.0, 0.0),
                        |(acc_self_sq, acc_other_sq, acc_self_other),
                         (row_self_sq, row_other_sq, row_self_other)| {
                            (
                                acc_self_sq + row_self_sq,
                                acc_other_sq + row_other_sq,
                                acc_self_other + row_self_other,
                            )
                        },
                    );
                Ok(acc_self_other / (acc_self_sq * acc_other_sq).sqrt())
            }
        }
    };
}

impl_rayon_corr!(i32, f32);
impl_rayon_corr!(f32, f32);
impl_rayon_corr!(f64, f64);

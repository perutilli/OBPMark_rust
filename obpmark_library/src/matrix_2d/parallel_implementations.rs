use crate::matrix_2d::Matrix2d;
use crate::number_traits::{Float, Number};
use crate::{parallel_traits::*, BaseMatrix, FirFilter};
use crate::{Error, Padding};

use std::sync::Arc;
use std::thread;

use crate::{Convolution, MatMul, MaxPooling, Relu, Softmax, LRN};

impl<T: Number> ParallelMatMul for Matrix2d<T> {
    fn parallel_multiply(
        &self,
        other: &Self,
        result: &mut Self,
        n_threads: usize,
    ) -> Result<(), Error> {
        if self.cols != other.rows || self.rows != result.rows || other.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }

        let rows_per_thread = (self.rows - 1) / n_threads + 1;

        let shared_self = Arc::new(self);
        let shared_other_transposed = Arc::new(other.transpose());

        thread::scope(|s| {
            result
                .data
                .chunks_mut(rows_per_thread)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let shared_self = shared_self.clone();
                    let shared_other = shared_other_transposed.clone();
                    let start_row = chunk_idx * rows_per_thread;
                    s.spawn(move || {
                        for (i, row) in chunk.iter_mut().enumerate() {
                            shared_self.multiply_row(&shared_other, row, start_row + i);
                        }
                    });
                });
        });

        Ok(())
    }
}

impl<T: Number> ParallelConvolution for Matrix2d<T> {
    fn parallel_convolute(
        &self,
        kernel: &Self,
        padding: Padding,
        result: &mut Self,
        n_threads: usize,
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

        if self.rows % n_threads != 0 {
            return Err(Error::InvalidNumberOfThreads);
        }

        let rows_per_thread = self.rows / n_threads;

        let shared_kernel = Arc::new(kernel);

        thread::scope(|s| {
            result
                .data
                .chunks_exact_mut(rows_per_thread)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let shared_kernel = shared_kernel.clone();
                    let start_row = chunk_idx * rows_per_thread;
                    s.spawn(move || {
                        for (i, row) in chunk.iter_mut().enumerate() {
                            self.convolute_row(&shared_kernel, row, start_row + i);
                        }
                    });
                });
        });

        Ok(())
    }
}

impl<T: Number> ParallelRelu for Matrix2d<T> {
    fn parallel_relu(&self, result: &mut Self, n_threads: usize) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }

        let rows_per_thread = (self.rows - 1) / n_threads + 1;

        thread::scope(|s| {
            result
                .data
                .chunks_mut(rows_per_thread)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let start_row = chunk_idx * rows_per_thread;
                    s.spawn(move || {
                        for (i, row) in chunk.iter_mut().enumerate() {
                            self.relu_row(row, start_row + i);
                        }
                    });
                });
        });

        Ok(())
    }
}

impl<T: Float> ParallelSoftmax for Matrix2d<T> {
    fn parallel_softmax(&self, result: &mut Self, n_threads: usize) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }

        let rows_per_thread = (self.rows - 1) / n_threads + 1;
        let mut total_sum = T::zero();

        thread::scope(|s| {
            result
                .data
                .chunks_mut(rows_per_thread)
                .enumerate()
                .map(|(chunk_idx, chunk)| {
                    let start_row = chunk_idx * rows_per_thread;
                    s.spawn(move || {
                        let mut sum = T::zero();
                        for (i, row) in chunk.iter_mut().enumerate() {
                            sum += self.softmax_row(row, start_row + i);
                        }
                        sum
                    })
                })
                .for_each(|handle| {
                    total_sum += handle.join().unwrap();
                });
        });

        thread::scope(|s| {
            result.data.chunks_mut(rows_per_thread).for_each(|chunk| {
                s.spawn(|| {
                    chunk.iter_mut().flatten().for_each(|el| {
                        *el /= total_sum;
                    });
                });
            });
        });

        Ok(())
    }
}

impl<T: Number> ParallelMaxPooling for Matrix2d<T> {
    fn parallel_max_pooling(
        &self,
        result: &mut Self,
        row_stride: usize,
        col_stride: usize,
        n_threads: usize,
    ) -> Result<(), Error> {
        if self.rows != result.rows * row_stride || self.cols != result.cols * col_stride {
            return Err(Error::InvalidDimensions);
        }

        let rows_per_thread = (self.rows - 1) / n_threads + 1;

        thread::scope(|s| {
            result
                .data
                .chunks_mut(rows_per_thread)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let start_row = chunk_idx * rows_per_thread;
                    s.spawn(move || {
                        for (i, row) in chunk.iter_mut().enumerate() {
                            self.max_pooling_row(row, start_row + i, row_stride, col_stride);
                        }
                    });
                });
        });

        Ok(())
    }
}

impl<T: Float> ParallelLRN<T> for Matrix2d<T> {
    fn parallel_lrn(
        &self,
        result: &mut Self,
        alpha: T,
        beta: T,
        k: T,
        n_threads: usize,
    ) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }

        let rows_per_thread = (self.rows - 1) / n_threads + 1;

        thread::scope(|s| {
            result
                .data
                .chunks_mut(rows_per_thread)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let start_row = chunk_idx * rows_per_thread;
                    s.spawn(move || {
                        for (i, row) in chunk.iter_mut().enumerate() {
                            self.lrn_row(row, start_row + i, alpha, beta, k);
                        }
                    });
                });
        });

        Ok(())
    }
}

impl<T: Number> ParallelFiniteImpulseResponseFilter for Matrix2d<T> {
    fn parallel_fir_filter(
        &self,
        kernel: &Self,
        result: &mut Self,
        n_threads: usize,
    ) -> Result<(), Error> {
        if self.rows != result.rows || result.cols != self.cols + kernel.cols - 1 || self.rows != 1
        {
            return Err(Error::InvalidDimensions);
        }
        if kernel.rows != 1 {
            return Err(Error::InvalidKernelDimensions);
        }

        // here the number of rows will always be one
        let elements_per_thread = (result.cols - 1) / n_threads + 1;

        let shared_kernel = Arc::new(kernel);

        thread::scope(|s| {
            result.data[0] // we only have one row, so this is all the data
                .chunks_mut(elements_per_thread)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let shared_kernel = shared_kernel.clone();
                    let start_idx = chunk_idx * elements_per_thread;
                    s.spawn(move || {
                        chunk.iter_mut().enumerate().for_each(|(i, result_el)| {
                            *result_el = self.fir_filter_element(&shared_kernel, i + start_idx);
                        });
                    });
                });
        });

        Ok(())
    }
}

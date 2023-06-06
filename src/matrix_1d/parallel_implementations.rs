use crate::matrix_1d::Matrix1d;
use crate::number_traits::Number;
use crate::parallel_traits::*;
use crate::{Error, Padding};

use std::sync::Arc;
use std::thread;

use crate::{Convolution, MatMul};

impl<T: Number> ParallelMatMul for Matrix1d<T> {
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
        let shared_other = Arc::new(other);

        thread::scope(|s| {
            result
                .data
                .chunks_mut(result.cols * rows_per_thread)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let shared_self = shared_self.clone();
                    let shared_other = shared_other.clone();
                    let start_row = chunk_idx * rows_per_thread;
                    s.spawn(move || {
                        chunk
                            .chunks_mut(shared_self.cols)
                            .enumerate()
                            .for_each(|(i, row)| {
                                shared_self.multiply_row(&shared_other, row, start_row + i);
                            });
                    });
                });
        });

        Ok(())
    }
}

impl<T: Number> ParallelConvolution for Matrix1d<T> {
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
                .chunks_exact_mut(rows_per_thread * self.cols)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let shared_kernel = shared_kernel.clone();
                    let start_row = chunk_idx * rows_per_thread;
                    s.spawn(move || {
                        for (i, row) in chunk.chunks_exact_mut(self.cols).enumerate() {
                            self.convolute_row(&shared_kernel, row, start_row + i);
                        }
                    });
                });
        });

        Ok(())
    }
}

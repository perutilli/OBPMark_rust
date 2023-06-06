use crate::matrix_1d::Matrix1d;
use crate::number_traits::Number;
use crate::parallel_traits::*;
use crate::{Error, Padding};

use std::sync::Arc;
use std::thread;

use crate::MatMul;

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

        let kernel_y_radius = (kernel.rows - 1) / 2;
        let kernel_x_radius = (kernel.cols - 1) / 2;

        let shared_self = Arc::new(self);
        let shared_kernel = Arc::new(kernel);

        thread::scope(|s| {
            result
                .data
                .chunks_exact_mut(rows_per_thread * shared_self.cols)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let shared_self = shared_self.clone();
                    let shared_kernel = shared_kernel.clone();
                    let start_row = chunk_idx * rows_per_thread;
                    s.spawn(move || {
                        for (i, row) in chunk.chunks_exact_mut(shared_self.cols).enumerate() {
                            for j in 0..shared_self.cols {
                                let mut sum = T::zero();
                                for k in 0..shared_kernel.rows {
                                    for l in 0..shared_kernel.cols {
                                        let y =
                                            (start_row + i + k) as isize - kernel_y_radius as isize;
                                        let x = (j + l) as isize - kernel_x_radius as isize;
                                        if (y > 0 && y < shared_self.rows as isize)
                                            && (x > 0 && x < shared_self.cols as isize)
                                        {
                                            let y = y as usize;
                                            let x = x as usize;
                                            sum += shared_self.data[y * shared_self.cols + x]
                                                * shared_kernel.data[k * shared_kernel.cols + l];
                                        }
                                    }
                                }
                                row[j] = sum;
                            }
                        }
                    });
                });
        });

        Ok(())
    }
}

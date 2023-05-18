use crate::matrix_1d::Matrix1d;
use crate::number_traits::Number;
use crate::parallel_traits::*;
use crate::Error;

use std::sync::Arc;
use std::thread;

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
        if self.rows % n_threads != 0 {
            return Err(Error::InvalidNumberOfThreads);
        }
        let rows_per_thread = self.rows / n_threads;

        let self_data = Arc::new(self);
        let other_data = Arc::new(other);
        let result_cols = result.cols;

        thread::scope(|s| {
            result
                .data
                .chunks_exact_mut(result.cols * rows_per_thread)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let self_data = self_data.clone();
                    let other_data = other_data.clone();
                    let start_row = chunk_idx * rows_per_thread;
                    s.spawn(move || {
                        for (j, el) in chunk.iter_mut().enumerate() {
                            let mut sum = T::zero();
                            for k in 0..self_data.cols {
                                sum += self_data.data
                                    [(start_row + j / result_cols) * self_data.cols + k]
                                    * other_data.data[k * other_data.cols + j % result_cols];
                            }
                            *el = sum;
                        }
                    });
                });
        });

        Ok(())
    }
}

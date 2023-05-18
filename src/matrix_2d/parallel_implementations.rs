use crate::matrix_2d::Matrix2d;
use crate::number_traits::Number;
use crate::parallel_traits::*;
use crate::Error;

use std::sync::Arc;
use std::thread;

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
        if self.rows % n_threads != 0 {
            return Err(Error::InvalidNumberOfThreads);
        }

        let rows_per_thread = self.rows / n_threads;

        let self_data = Arc::new(self);
        let other_data = Arc::new(other);

        thread::scope(|s| {
            result
                .data
                .chunks_exact_mut(rows_per_thread)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let self_data = self_data.clone();
                    let other_data = other_data.clone();
                    let start_row = chunk_idx * rows_per_thread;
                    s.spawn(move || {
                        for (i, row) in chunk.iter_mut().enumerate() {
                            for (j, el) in row.iter_mut().enumerate() {
                                let mut sum = T::zero();
                                for k in 0..self_data.cols {
                                    sum += self_data.data[start_row + i][k] * other_data.data[k][j];
                                }
                                *el = sum;
                            }
                        }
                    });
                });
        });

        Ok(())
    }
}

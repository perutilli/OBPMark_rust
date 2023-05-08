use std::sync::Arc;
use std::thread;

use crate::{format_number, BaseMatrix, Error, Number, ParallelMatMul};

pub struct Matrix2d<T: Number> {
    data: Vec<Vec<T>>,
    rows: usize,
    cols: usize,
}

impl<T: Number> BaseMatrix<T> for Matrix2d<T> {
    fn new(data: Vec<Vec<T>>, rows: usize, cols: usize) -> Matrix2d<T> {
        Matrix2d { data, rows, cols }
    }

    fn get_data(&self) -> Vec<Vec<T>> {
        self.data.clone()
    }

    fn reshape(&mut self, new_rows: usize, new_cols: usize) -> Result<(), Error> {
        println!("WARNING: reshape is an expensive operation for 2d matrices");
        if self.rows * self.cols != new_rows * new_cols {
            return Err(Error::InvalidDimensions);
        }
        let old_data = self.data.clone();
        let flat_data: Vec<_> = old_data.into_iter().flatten().collect();
        self.data = flat_data
            .chunks(new_cols)
            .map(|chunk| chunk.to_vec())
            .collect();
        self.rows = new_rows;
        self.cols = new_cols;
        Ok(())
    }
}

impl_display!(Matrix2d);

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
        // first we create an Arc for the input data
        let self_data = Arc::new(self.clone());
        let other_data = Arc::new(other.clone());
        thread::scope(|s| {
            let mut threads_handles = Vec::new();
            let rows_per_thread = self.rows / n_threads;
            // this scope says that all the threads will be joined before the scope ends
            // i.e right after the most external for loop ends
            // the compiler is not able to infer the lifetime of the threads
            for chunk in 0..(self.rows / rows_per_thread) {
                let self_data = self_data.clone();
                let other_data = other_data.clone();
                //let result_row = &mut (result.data[i]);
                threads_handles.push(s.spawn(move || {
                    let mut result_rows = vec![vec![T::zero(); other.cols]; rows_per_thread];
                    for i in 0..rows_per_thread {
                        for j in 0..other.cols {
                            let mut sum = T::zero();
                            for k in 0..self.cols {
                                sum += self_data.data[i + chunk * rows_per_thread][k]
                                    * other_data.data[k][j];
                            }
                            result_rows[i][j] = sum;
                        }
                    }
                    result_rows
                }));
            }
            for (i, handle) in threads_handles.into_iter().enumerate() {
                let chunk = handle.join().unwrap();
                for (j, row) in chunk.into_iter().enumerate() {
                    result.data[i * rows_per_thread + j] = row;
                }
            }
        });

        Ok(())
    }
}

mod rayon_implementations;
mod sequential_implementations;

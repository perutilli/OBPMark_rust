use funty::Floating;
use std::sync::Arc;
use std::thread;

use crate::{
    format_number, BaseMatrix, Convolution, Correlation, Error, FastFourierTransform, MatMul,
    MaxPooling, Num, ParallelMatMul, Relu, Softmax, LRN,
};

pub struct Matrix2d<T: Num> {
    data: Vec<Vec<T>>,
    rows: usize,
    cols: usize,
}

impl<T: Num> BaseMatrix<T> for Matrix2d<T> {
    fn new(data: Vec<Vec<T>>, rows: usize, cols: usize) -> Matrix2d<T> {
        Matrix2d { data, rows, cols }
    }

    fn get_data(&self) -> Vec<Vec<T>> {
        self.data.clone()
    }

    fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.rows && col < self.cols, "Invalid indexing");
        self.data[row][col] = value;
    }
}

impl_display!(Matrix2d);

impl<T: Num> MatMul for Matrix2d<T> {
    fn multiply(&self, other: &Matrix2d<T>, result: &mut Matrix2d<T>) -> Result<(), Error> {
        if self.cols != other.rows {
            return Err(Error::InvalidDimensions);
        }

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::zero();
                // NOTE: this allows result to not be all zeros
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        Ok(())
    }
}

impl<T: Num> Relu for Matrix2d<T> {
    fn relu(&self, result: &mut Matrix2d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i][j] < T::zero() {
                    result.data[i][j] = T::zero();
                } else {
                    result.data[i][j] = self.data[i][j];
                }
                // result.data[i][j] = self.data[i][j].max(T::default());
            }
        }
        Ok(())
    }
}

impl<T: Num + Floating> Softmax for Matrix2d<T> {
    fn softmax(&self, result: &mut Matrix2d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            let mut sum = T::zero();
            for j in 0..self.cols {
                let val = self.data[i][j].exp();
                sum += val;
                result.data[i][j] = val;
            }
            for j in 0..self.cols {
                result.data[i][j] /= sum;
            }
        }
        Ok(())
    }
}

impl<T: Num> MaxPooling for Matrix2d<T> {
    fn max_pooling(
        &self,
        result: &mut Matrix2d<T>,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<(), Error> {
        if self.rows != result.rows * row_stride || self.cols != result.cols * col_stride {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..result.rows {
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
                result.data[i][j] = max;
            }
        }
        Ok(())
    }
}

impl<T: Num> Correlation for Matrix2d<T> {
    fn correlation(&self, other: &Matrix2d<T>) -> Result<f64, Error> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(Error::InvalidDimensions);
        }

        let mut acc_self_sq = 0_f64;
        let mut acc_other_sq = 0_f64;
        let mut acc_self_other = 0_f64;

        let self_mean =
            self.data.iter().flatten().sum::<T>().as_f64() / (self.rows * self.cols) as f64;
        let other_mean =
            other.data.iter().flatten().sum::<T>().as_f64() / (other.rows * other.cols) as f64;

        for i in 0..self.rows {
            for j in 0..self.cols {
                let self_delta = self.data[i][j].as_f64() - self_mean;
                let other_delta = other.data[i][j].as_f64() - other_mean;
                acc_self_sq += self_delta * self_delta;
                acc_other_sq += other_delta * other_delta;
                acc_self_other += self_delta * other_delta;
            }
        }
        Ok(acc_self_other / (acc_self_sq * acc_other_sq).sqrt())
    }
}

use crate::Padding;
impl<T: Num> Convolution for Matrix2d<T> {
    fn convolute(&self, kernel: &Self, padding: Padding, result: &mut Self) -> Result<(), Error> {
        match padding {
            Padding::Zeroes => (),
        }

        if self.rows != result.rows || self.cols != result.cols {
            // NOTE: this is a very specific kind of convolution, we probably want to support
            //       more general cases, in particular at least the no padding case
            return Err(Error::InvalidDimensions);
        }

        if kernel.rows % 2 == 0 || kernel.cols % 2 == 0 {
            return Err(Error::InvalidKernelDimensions);
        }

        let kernel_y_radius = (kernel.rows - 1) / 2;
        let kernel_x_radius = (kernel.cols - 1) / 2;

        for i in 0..self.rows {
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
                result.data[i][j] = sum;
            }
        }
        Ok(())
    }
}

impl<T: Num + Floating> LRN<T> for Matrix2d<T> {
    fn lrn(&self, result: &mut Self, alpha: T, beta: T, k: T) -> Result<(), Error> {
        // TODO: this is actually a special case where n = 1, ok for the benchmark but not general
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] =
                    self.data[i][j] / (k + alpha * self.data[i][j] * self.data[i][j]).powf(beta);
            }
        }
        Ok(())
    }
}

macro_rules! impl_fft {
    ($t:tt) => {
        impl FastFourierTransform for Matrix2d<$t> {
            fn fft(&mut self, nn: usize) -> Result<(), Error> {
                if self.rows != 1 {
                    return Err(Error::InvalidDimensions);
                }

                let data = &mut self.data[1];

                let n = nn << 1;
                let mut j = 1;
                for i in 1..n {
                    if j > i {
                        data.swap(j - 1, i - 1);
                        data.swap(j, i);
                    }
                    let mut m = nn;
                    while m >= 2 && j > m {
                        j -= m;
                        m >>= 1;
                    }
                    j += m;
                }

                let mut mmax = 2;
                while n > mmax {
                    let istep = mmax << 1;
                    let theta = -(2.0 * std::$t::consts::PI / mmax as $t);
                    let wtemp = (theta / 2.0).sin();
                    let wpr = -2.0 * wtemp * wtemp;
                    let wpi = (theta / 2.0).sin();
                    let mut wr = 1.0;
                    let mut wi = 0.0;
                    for m in (1..mmax).step_by(2) {
                        for i in (m..=n).step_by(istep) {
                            let j = i + mmax;
                            let tempr = wr * data[j - 1] - wi * data[j];
                            let tempi = wr * data[j] + wi * data[j - 1];
                            data[j - 1] = data[i - 1] - tempr;
                            data[j] = data[i] - tempi;
                            data[i - 1] += tempr;
                            data[i] += tempi;
                        }
                        let wtemp = wr;
                        wr += wr * wpr - wi * wpi;
                        wi += wi * wpr + wtemp * wpi;
                    }
                    mmax = istep;
                }
                Ok(())
            }
        }
    };
    () => {};
}

impl_fft!(f32);
impl_fft!(f64);

impl<T: Num> ParallelMatMul for Matrix2d<T> {
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

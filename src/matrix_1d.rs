use std::sync::Arc;
use std::thread;

use crate::{
    format_number, BaseMatrix, Convolution, Correlation, Error, FastFourierTransform,
    FastFourierTransformWindowed, Float, MatMul, MaxPooling, Number, ParallelMatMul, Relu, Softmax,
    WaveletTransformFloating, WaveletTransformInteger, LRN,
};

pub struct Matrix1d<T: Number> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Number> BaseMatrix<T> for Matrix1d<T> {
    fn new(data: Vec<Vec<T>>, rows: usize, cols: usize) -> Matrix1d<T> {
        Matrix1d {
            data: data.into_iter().flatten().collect(),
            rows,
            cols,
        }
    }

    fn get_data(&self) -> Vec<Vec<T>> {
        self.data
            .clone()
            .chunks(self.cols)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<T>>>()
    }

    fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.rows && col < self.cols, "Invalid indexing");
        self.data[row * self.cols + col] = value;
    }

    fn reshape(&mut self, new_rows: usize, new_cols: usize) -> Result<(), Error> {
        if new_rows * new_cols != self.rows * self.cols {
            return Err(Error::InvalidDimensions);
        }
        self.rows = new_rows;
        self.cols = new_cols;
        Ok(())
    }
}

impl_display!(Matrix1d);

impl<T: Number> MatMul for Matrix1d<T> {
    fn multiply(&self, other: &Matrix1d<T>, result: &mut Matrix1d<T>) -> Result<(), Error> {
        if self.cols != other.rows || self.rows != result.rows || other.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::zero();
                // NOTE: this allows result to not be all zeros
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * other.cols + j] = sum;
            }
        }
        Ok(())
    }
}
/*
impl MatMul for Matrix1d<f16> {
    expand_multiply!(f16, f16::from_f32(0.0));
}
*/

impl<T: Number> Relu for Matrix1d<T> {
    fn relu(&self, result: &mut Matrix1d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i * self.cols + j] > T::zero() {
                    result.data[i * self.cols + j] = self.data[i * self.cols + j];
                } else {
                    result.data[i * self.cols + j] = T::zero();
                }
                // result.data[i * self.rows + j] = self.data[i * self.rows + j].max(T::default());
            }
        }
        Ok(())
    }
}

impl<T: Number + num_traits::Float> Softmax for Matrix1d<T> {
    fn softmax(&self, result: &mut Matrix1d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            let mut sum = T::zero();
            for j in 0..self.cols {
                let val = self.data[i * self.cols + j].exp();
                sum += val;
                result.data[i * self.cols + j] = val;
            }
            for j in 0..self.cols {
                result.data[i * self.cols + j] /= sum;
            }
        }
        Ok(())
    }
}

impl<T: Number> MaxPooling for Matrix1d<T> {
    fn max_pooling(
        &self,
        result: &mut Matrix1d<T>,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<(), Error> {
        if self.rows != result.rows * row_stride || self.cols != result.cols * col_stride {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..result.rows {
            for j in 0..result.cols {
                let mut max = self.data[i * row_stride * self.cols + j * col_stride];
                for k in 0..row_stride {
                    for l in 0..col_stride {
                        if max
                            < self.data
                                [i * row_stride * self.cols + j * col_stride + k * self.cols + l]
                        {
                            max = self.data
                                [i * row_stride * self.cols + j * col_stride + k * self.cols + l];
                        }
                        /*
                        max = max.max(
                            self.data
                                [i * row_stride * self.cols + j * col_stride + k * self.cols + l],
                        );
                         */
                    }
                }
                result.data[i * result.cols + j] = max;
            }
        }
        Ok(())
    }
}

impl<T: Number> Correlation for Matrix1d<T> {
    fn correlation(&self, other: &Matrix1d<T>) -> Result<f64, Error> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(Error::InvalidDimensions);
        }

        let mut acc_self_sq = 0_f64;
        let mut acc_other_sq = 0_f64;
        let mut acc_self_other = 0_f64;

        let self_mean = self.data.iter().sum::<T>().as_() / (self.rows * self.cols) as f64;
        let other_mean = other.data.iter().sum::<T>().as_() / (other.rows * other.cols) as f64;

        for i in 0..self.rows {
            for j in 0..self.cols {
                let self_delta = self.data[i * self.cols + j].as_() - self_mean;
                let other_delta = other.data[i * self.cols + j].as_() - other_mean;
                acc_self_sq += self_delta * self_delta;
                acc_other_sq += other_delta * other_delta;
                acc_self_other += self_delta * other_delta;
            }
        }
        Ok(acc_self_other / (acc_self_sq * acc_other_sq).sqrt())
    }
}

use crate::Padding;
impl<T: Number> Convolution for Matrix1d<T> {
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
                            sum += self.data[y * self.cols + x] * kernel.data[k * kernel.cols + l];
                        }
                    }
                }
                result.data[i * result.cols + j] = sum;
            }
        }
        Ok(())
    }
}

impl<T: Float> LRN<T> for Matrix1d<T> {
    fn lrn(&self, result: &mut Self, alpha: T, beta: T, k: T) -> Result<(), Error> {
        // TODO: this is actually a special case where n = 1, ok for the benchmark but not general
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i * result.cols + j] = self.data[i * self.cols + j]
                    / (k + alpha * self.data[i * self.cols + j] * self.data[i * self.cols + j])
                        .powf(beta);
            }
        }
        Ok(())
    }
}

macro_rules! impl_fft {
    ($t:tt) => {
        impl FastFourierTransform for Matrix1d<$t> {
            fn fft(&mut self, nn: usize, start_pos: usize) -> Result<(), Error> {
                if self.rows != 1 {
                    return Err(Error::InvalidDimensions);
                }

                let window = nn << 1;

                let n = nn << 1;
                let mut j = 1;
                for i in (1..n).step_by(2) {
                    if j > i {
                        self.data
                            .swap(window * start_pos + j - 1, window * start_pos + i - 1);
                        self.data
                            .swap(window * start_pos + j, window * start_pos + i);
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
                    let wpi = (theta).sin();
                    let mut wr = 1.0;
                    let mut wi = 0.0;
                    for m in (1..mmax).step_by(2) {
                        for i in (m..=n).step_by(istep) {
                            let j = i + mmax;
                            let tempr = wr * self.data[window * start_pos + j - 1]
                                - wi * self.data[window * start_pos + j];
                            let tempi = wr * self.data[window * start_pos + j]
                                + wi * self.data[window * start_pos + j - 1];
                            self.data[window * start_pos + j - 1] =
                                self.data[window * start_pos + i - 1] - tempr;
                            self.data[window * start_pos + j] =
                                self.data[window * start_pos + i] - tempi;
                            self.data[window * start_pos + i - 1] += tempr;
                            self.data[window * start_pos + i] += tempi;
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

impl FastFourierTransformWindowed for Matrix1d<f32> {
    fn fftw(&mut self, nn: usize, window: usize, result: &mut Self) -> Result<(), Error> {
        if self.rows != 1 {
            return Err(Error::InvalidDimensions);
        }
        for i in (0..(nn * 2 - window + 1)).step_by(2) {
            for j in 0..window {
                result.data[i * window + j] = self.data[i + j];
            }
            result.fft(window >> 1, i)?;
        }
        Ok(())
    }
}

// TODO: code duplication, turn into a macro
impl FastFourierTransformWindowed for Matrix1d<f64> {
    fn fftw(&mut self, nn: usize, window: usize, result: &mut Self) -> Result<(), Error> {
        if self.rows != 1 {
            return Err(Error::InvalidDimensions);
        }
        for i in (0..(nn * 2 - window + 1)).step_by(2) {
            for j in 0..window {
                result.data[i * window + j] = self.data[i + j];
            }
            result.fft(window >> 1, i)?;
        }
        Ok(())
    }
}

// TODO: note that right now the data has a minimum size for the algorithm to work
//       should at least document this in the error
impl WaveletTransformInteger<i32> for Matrix1d<i32> {
    fn wavelet_transform(&self, result: &mut Self, size: usize) -> Result<(), Error> {
        let full_size = size * 2;
        if self.rows != 1 || self.cols != full_size {
            return Err(Error::InvalidDimensions);
        }

        let data = &self.data;

        // high part
        for i in 0..size {
            result.data[i + size] = if i == 0 {
                data[1]
                    - (((9.0 / 16.0) * (data[0] + data[2]) as f32)
                        - ((1.0 / 16.0) * (data[2] + data[4]) as f32)
                        + (1.0 / 2.0)) as i32
            } else if i == size - 2 {
                data[2 * size - 3]
                    - (((9.0 / 16.0) * (data[2 * size - 4] + data[2 * size - 2]) as f32)
                        - ((1.0 / 16.0) * (data[2 * size - 6] + data[2 * size - 2]) as f32)
                        + (1.0 / 2.0)) as i32
            } else if i == size - 1 {
                data[2 * size - 1]
                    - (((9.0 / 8.0) * (data[2 * size - 2]) as f32)
                        - ((1.0 / 8.0) * (data[2 * size - 4]) as f32)
                        + (1.0 / 2.0)) as i32
            } else {
                data[2 * i + 1]
                    - (((9.0 / 16.0) * (data[2 * i] + data[2 * i + 2]) as f32)
                        - ((1.0 / 16.0) * (data[2 * i - 2] + data[2 * i + 4]) as f32)
                        + (1.0 / 2.0)) as i32
            };
        }

        // low part
        for i in 0..size {
            result.data[i] = if i == 0 {
                data[0] - (-result.data[size] / 2 + 1)
            } else {
                data[2 * i] - (-((result.data[i + size - 1] + result.data[i + size]) / 4) + 1)
            };
        }

        Ok(())
    }
}

impl<T: Float> WaveletTransformFloating<T> for Matrix1d<T> {
    fn wavelet_transform(
        &self,
        result: &mut Self,
        size: usize,
        low_pass_filter: &[T],
        low_pass_filter_size: usize,
        high_pass_filter: &[T],
        high_pass_filter_size: usize,
    ) -> Result<(), Error> {
        let full_size = size * 2;

        if self.rows != 1 || self.cols != full_size {
            return Err(Error::InvalidDimensions);
        }

        let hi_start = -(low_pass_filter_size as isize / 2);
        let hi_end = (low_pass_filter_size / 2) as isize;
        let gi_start = -(high_pass_filter_size as isize / 2);
        let gi_end = (high_pass_filter_size / 2) as isize;

        for i in 0..size {
            let mut sum_value_low = T::zero();
            // process the lowpass filter
            for hi in hi_start..hi_end + 1 {
                let x_position = (2 * i) as isize + hi;
                let x_position = if x_position < 0 {
                    x_position * -1
                } else if x_position > full_size as isize - 1 {
                    full_size as isize - 1 - (x_position - (full_size as isize - 1))
                } else {
                    x_position
                };
                sum_value_low +=
                    low_pass_filter[(hi + hi_end) as usize] * self.data[x_position as usize];
            }
            result.data[i] = sum_value_low;

            let mut sum_value_high = T::zero();
            // process the highpass filter
            for gi in gi_start..gi_end + 1 {
                let x_position = (2 * i) as isize + gi + 1;
                let x_position = if x_position < 0 {
                    x_position * -1
                } else if x_position > full_size as isize - 1 {
                    full_size as isize - 1 - (x_position - (full_size as isize - 1))
                } else {
                    x_position
                };
                sum_value_high +=
                    high_pass_filter[(gi + gi_end) as usize] * self.data[x_position as usize];
            }
            result.data[i + size] = sum_value_high;
        }
        Ok(())
    }
}

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
                    let mut result_rows = vec![T::zero(); other.cols * rows_per_thread];
                    for i in 0..rows_per_thread {
                        for j in 0..other.cols {
                            let mut sum = T::zero();
                            for k in 0..self.cols {
                                sum += self_data.data
                                    [(i + chunk * rows_per_thread) * self.cols + k]
                                    * other_data.data[k * other.cols + j];
                            }
                            result_rows[i * other.cols + j] = sum;
                        }
                    }
                    result_rows
                }));
            }
            for (i, handle) in threads_handles.into_iter().enumerate() {
                let chunk = handle.join().unwrap();
                for (j, el) in chunk.into_iter().enumerate() {
                    result.data[i * result.cols * rows_per_thread + j] = el;
                }
            }
        });

        Ok(())
    }
}

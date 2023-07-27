use super::Matrix1d;
use crate::{
    Convolution, Correlation, Error, FastFourierTransform, FastFourierTransformWindowed, Float,
    MatMul, MaxPooling, Number, Relu, Softmax, WaveletTransformFloating, WaveletTransformInteger,
    LRN,
};

impl<T: Number> MatMul<T> for Matrix1d<T> {
    fn multiply_row(&self, other: &Matrix1d<T>, result_row: &mut [T], row_idx: usize) {
        let i = row_idx;
        for j in 0..other.cols {
            let mut sum = T::zero();
            for k in 0..self.cols {
                sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
            }
            result_row[j] = sum; // note that j is already the position in the chunk
        }
    }

    fn multiply(&self, other: &Matrix1d<T>, result: &mut Matrix1d<T>) -> Result<(), Error> {
        if self.cols != other.rows || self.rows != result.rows || other.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .chunks_exact_mut(self.cols)
            .enumerate()
            .for_each(|(i, result_row)| self.multiply_row(other, result_row, i));
        Ok(())
    }
}

impl<T: Number> Relu<T> for Matrix1d<T> {
    fn relu_row(&self, result_row: &mut [T], row_idx: usize) {
        let i = row_idx;
        for j in 0..self.cols {
            if self.data[i * self.cols + j] > T::zero() {
                result_row[j] = self.data[i * self.cols + j];
            } else {
                result_row[j] = T::zero();
            }
        }
    }

    fn relu(&self, result: &mut Matrix1d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .chunks_mut(self.cols)
            .enumerate()
            .for_each(|(i, result_row)| self.relu_row(result_row, i));
        Ok(())
    }
}

impl<T: Float> Softmax<T> for Matrix1d<T> {
    fn softmax_row(&self, result_row: &mut [T], row_idx: usize) {
        let i = row_idx;
        let mut sum = T::zero();
        for j in 0..self.cols {
            let val = self.data[i * self.cols + j].exp();
            sum += val;
            result_row[j] = val;
        }
        for j in 0..self.cols {
            result_row[j] /= sum;
        }
    }

    fn softmax(&self, result: &mut Matrix1d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .chunks_mut(self.cols)
            .enumerate()
            .for_each(|(i, result_row)| self.softmax_row(result_row, i));
        Ok(())
    }
}

impl<T: Number> MaxPooling<T> for Matrix1d<T> {
    fn max_pooling_row(
        &self,
        result_row: &mut [T],
        row_idx: usize,
        row_stride: usize,
        col_stride: usize,
    ) {
        let i = row_idx;
        for j in 0..result_row.len() {
            let mut max = self.data[i * row_stride * self.cols + j * col_stride];
            for k in 0..row_stride {
                for l in 0..col_stride {
                    if max
                        < self.data[i * row_stride * self.cols + j * col_stride + k * self.cols + l]
                    {
                        max = self.data
                            [i * row_stride * self.cols + j * col_stride + k * self.cols + l];
                    }
                }
            }
            result_row[j] = max;
        }
    }

    fn max_pooling(
        &self,
        result: &mut Matrix1d<T>,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<(), Error> {
        if self.rows != result.rows * row_stride || self.cols != result.cols * col_stride {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .chunks_mut(result.cols)
            .enumerate()
            .for_each(|(i, result_row)| {
                self.max_pooling_row(result_row, i, row_stride, col_stride)
            });
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
impl<T: Number> Convolution<T> for Matrix1d<T> {
    fn convolute_row(&self, kernel: &Self, result_row: &mut [T], row_idx: usize) {
        let i = row_idx;
        let kernel_y_radius = (kernel.rows - 1) / 2;
        let kernel_x_radius = (kernel.cols - 1) / 2;
        for j in 0..result_row.len() {
            let mut sum = T::zero();
            for k in 0..kernel.rows {
                for l in 0..kernel.cols {
                    let y = (i + k) as isize - kernel_y_radius as isize;
                    let x = (j + l) as isize - kernel_x_radius as isize;
                    if (y > 0 && y < self.rows as isize) && (x > 0 && x < self.cols as isize) {
                        sum += self.data[y as usize * self.cols + x as usize]
                            * kernel.data[k * kernel.cols + l];
                    }
                }
            }
            result_row[j] = sum;
        }
    }

    fn convolute(&self, kernel: &Self, padding: Padding, result: &mut Self) -> Result<(), Error> {
        match padding {
            Padding::Zeroes => (),
        }

        if self.rows != result.rows || self.cols != result.cols {
            // NOTE: we only support this kind of convolution for the benchmarks
            return Err(Error::InvalidDimensions);
        }

        if kernel.rows % 2 == 0 || kernel.cols % 2 == 0 {
            return Err(Error::InvalidKernelDimensions);
        }

        result
            .data
            .chunks_mut(result.cols)
            .enumerate()
            .for_each(|(i, result_row)| self.convolute_row(kernel, result_row, i));
        Ok(())
    }
}

impl<T: Float> LRN<T> for Matrix1d<T> {
    fn lrn_row(&self, result_row: &mut [T], row_idx: usize, alpha: T, beta: T, k: T) {
        let i = row_idx;
        for j in 0..self.cols {
            result_row[j] = self.data[i * self.cols + j]
                / (k + alpha * self.data[i * self.cols + j] * self.data[i * self.cols + j])
                    .powf(beta);
        }
    }
    fn lrn(&self, result: &mut Self, alpha: T, beta: T, k: T) -> Result<(), Error> {
        // TODO: this is actually a special case where n = 1, ok for the benchmark but not general
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .chunks_mut(result.cols)
            .enumerate()
            .for_each(|(i, result_row)| self.lrn_row(result_row, i, alpha, beta, k));
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

#[cfg(test)]
mod tests {
    use super::super::BaseMatrix;
    #[allow(unused_imports)]
    use super::*;

    fn gen_matrix_data(size: usize) -> Vec<Vec<f32>> {
        let mut data = vec![vec![0.0; size]; size];
        for i in 0..size {
            for j in 0..size {
                unsafe {
                    data[i][j] = libc::rand() as f32 / 2147483647.0;
                }
            }
        }
        data
    }

    // TODO: this should be a trait called Signed maybe
    fn abs<T: Number + std::ops::Neg<Output = T>>(x: T) -> T {
        if x < T::zero() {
            -x
        } else {
            x
        }
    }

    #[test]
    fn correlation() {
        let m1;
        let m2;
        unsafe {
            libc::srand(21121993);
            m1 = Matrix1d::new(gen_matrix_data(4), 4, 4);
            m2 = Matrix1d::new(gen_matrix_data(4), 4, 4);
        }
        let res = m1.correlation(&m2).unwrap();
        assert!(abs(res - 0.073827) < 10e-6, "res = {}", res);
    }
}

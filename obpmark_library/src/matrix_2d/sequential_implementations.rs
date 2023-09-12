use super::Matrix2d;
use crate::{
    BaseMatrix, Convolution, Correlation, Error, FirFilter, Float, MatMul, MaxPooling, Number,
    Relu, Softmax, WaveletTransformFloating, WaveletTransformInteger, LRN,
};

impl<T: Number> MatMul<T> for Matrix2d<T> {
    fn multiply_row(&self, other: &Matrix2d<T>, result_row: &mut [T], row_idx: usize) {
        let i = row_idx;
        for j in 0..other.cols {
            let mut sum = T::zero();
            for k in 0..self.cols {
                sum += self.data[i][k] * other.data[j][k];
            }
            result_row[j] = sum; // note that j is already the position in the chunk
        }
    }

    fn multiply(&self, other: &Matrix2d<T>, result: &mut Matrix2d<T>) -> Result<(), Error> {
        if self.cols != other.rows || self.rows != result.rows || other.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }

        let other_transposed = other.transpose();

        result
            .data
            .iter_mut()
            .enumerate()
            .for_each(|(i, result_row)| self.multiply_row(&other_transposed, result_row, i));
        Ok(())
    }
}

impl<T: Number> Relu<T> for Matrix2d<T> {
    fn relu_row(&self, result_row: &mut [T], row_idx: usize) {
        let i = row_idx;
        for j in 0..self.cols {
            if self.data[i][j] < T::zero() {
                result_row[j] = T::zero();
            } else {
                result_row[j] = self.data[i][j];
            }
        }
    }
    fn relu(&self, result: &mut Matrix2d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .iter_mut()
            .enumerate()
            .for_each(|(i, result_row)| self.relu_row(result_row, i));
        Ok(())
    }
}

impl<T: Number + num_traits::Float> Softmax<T> for Matrix2d<T> {
    fn softmax_row(&self, result_row: &mut [T], row_idx: usize) -> T {
        let i = row_idx;
        let mut sum = T::zero();
        for j in 0..self.cols {
            let val = self.data[i][j].exp();
            sum += val;
            result_row[j] = val;
        }
        sum
    }

    fn softmax(&self, result: &mut Matrix2d<T>) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        let mut sum = T::zero();

        result
            .data
            .iter_mut()
            .enumerate()
            .for_each(|(i, result_row)| sum += self.softmax_row(result_row, i));

        result
            .data
            .iter_mut()
            .flatten()
            .for_each(|el| *el = *el / sum);
        Ok(())
    }
}

impl<T: Number> MaxPooling<T> for Matrix2d<T> {
    fn max_pooling_row(
        &self,
        result_row: &mut [T],
        row_idx: usize,
        row_stride: usize,
        col_stride: usize,
    ) {
        let i = row_idx;
        for j in 0..result_row.len() {
            let mut max = self.data[i * row_stride][j * col_stride];
            for k in 0..row_stride {
                for l in 0..col_stride {
                    if max < self.data[i * row_stride + k][j * col_stride + l] {
                        max = self.data[i * row_stride + k][j * col_stride + l];
                    }
                }
            }
            result_row[j] = max;
        }
    }
    fn max_pooling(
        &self,
        result: &mut Matrix2d<T>,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<(), Error> {
        if self.rows != result.rows * row_stride || self.cols != result.cols * col_stride {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .iter_mut()
            .enumerate()
            .for_each(|(i, result_row)| {
                self.max_pooling_row(result_row, i, row_stride, col_stride)
            });
        Ok(())
    }
}

macro_rules! impl_correlation {
    ($self_type: tt, $output_type: tt) => {
        impl Correlation for Matrix2d<$self_type> {
            type Output = $output_type;
            fn accumulate_row(
                &self,
                other: &Self,
                self_mean: Self::Output,
                other_mean: Self::Output,
                row_idx: usize,
            ) -> (Self::Output, Self::Output, Self::Output) {
                let i = row_idx;
                let mut acc_self_sq = 0.0;
                let mut acc_other_sq = 0.0;
                let mut acc_self_other = 0.0;
                for j in 0..self.cols {
                    let self_diff = self.data[i][j] as $output_type - self_mean;
                    let other_diff = other.data[i][j] as $output_type - other_mean;
                    acc_self_sq += self_diff * self_diff;
                    acc_other_sq += other_diff * other_diff;
                    acc_self_other += self_diff * other_diff;
                }
                (acc_self_sq, acc_other_sq, acc_self_other)
            }

            fn correlation(&self, other: &Self) -> Result<Self::Output, Error> {
                if self.rows != other.rows || self.cols != other.cols {
                    return Err(Error::InvalidDimensions);
                }

                let mut acc_self_sq = 0.0;
                let mut acc_other_sq = 0.0;
                let mut acc_self_other = 0.0;

                let self_mean = self.data.iter().flatten().sum::<$self_type>() as $output_type
                    / (self.rows * self.cols) as $output_type;
                let other_mean = other.data.iter().flatten().sum::<$self_type>() as $output_type
                    / (other.rows * other.cols) as $output_type;

                for i in 0..self.rows {
                    let (row_self_sq, row_other_sq, row_self_other) =
                        self.accumulate_row(&other, self_mean, other_mean, i);
                    acc_self_sq += row_self_sq;
                    acc_other_sq += row_other_sq;
                    acc_self_other += row_self_other;
                }
                Ok(acc_self_other / (acc_self_sq * acc_other_sq).sqrt())
            }
        }
    };
}

impl_correlation!(i32, f32);
impl_correlation!(f32, f32);
impl_correlation!(f64, f64);

use crate::Padding;
impl<T: Number> Convolution<T> for Matrix2d<T> {
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
                    if (y >= 0 && y < self.rows as isize) && (x >= 0 && x < self.cols as isize) {
                        let y = y as usize;
                        let x = x as usize;
                        sum += self.data[y][x] * kernel.data[k][l];
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
            // NOTE: this is a very specific kind of convolution, we probably want to support
            //       more general cases, in particular at least the no padding case
            return Err(Error::InvalidDimensions);
        }

        if kernel.rows % 2 == 0 || kernel.cols % 2 == 0 {
            return Err(Error::InvalidKernelDimensions);
        }

        result
            .data
            .iter_mut()
            .enumerate()
            .for_each(|(i, result_row)| self.convolute_row(kernel, result_row, i));

        Ok(())
    }
}

impl<T: Float> LRN<T> for Matrix2d<T> {
    fn lrn_row(&self, result_row: &mut [T], row_idx: usize, alpha: T, beta: T, k: T) {
        let i = row_idx;
        for j in 0..self.cols {
            result_row[j] =
                self.data[i][j] / (k + alpha * self.data[i][j] * self.data[i][j]).powf(beta);
        }
    }
    fn lrn(&self, result: &mut Self, alpha: T, beta: T, k: T) -> Result<(), Error> {
        // TODO: this is actually a special case where n = 1, ok for the benchmark but not general
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .iter_mut()
            .enumerate()
            .for_each(|(i, result_row)| self.lrn_row(result_row, i, alpha, beta, k));
        Ok(())
    }
}

// TODO: note that right now the data has a minimum size for the algorithm to work
//       should at least document this in the error
impl WaveletTransformInteger<i32> for Matrix2d<i32> {
    fn wavelet_transform_bottom_half_element(
        &self,
        result_top_half: &[i32],
        element_idx: usize,
    ) -> i32 {
        let i = element_idx;
        let data = &self.data[0];

        if i == 0 {
            data[0] - (-(result_top_half[0] as f32 / 2.0) + 0.5) as i32
        } else {
            data[2 * i]
                - (-((result_top_half[i - 1] + result_top_half[i]) as f32 / 4.0) + 0.5) as i32
        }
    }
    fn wavelet_transform_top_half_element(&self, element_idx: usize, size: usize) -> i32 {
        let i = element_idx;
        let data = &self.data[0];

        if i == 0 {
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
        }
    }

    fn wavelet_transform(&self, result: &mut Self, size: usize) -> Result<(), Error> {
        let full_size = size * 2;
        if self.rows != 1 || self.cols != full_size {
            return Err(Error::InvalidDimensions);
        }

        // top half
        result.data[0] // 1D vector
            .iter_mut()
            .skip(size) // second half
            .enumerate()
            .for_each(|(idx, el)| {
                *el = self.wavelet_transform_top_half_element(idx, size);
            });

        let binding = result.data[0][size..].to_owned();
        let top_half = binding.as_slice();

        // bottom half
        result.data[0] // 1D vector
            .iter_mut()
            .take(size) // first half
            .enumerate()
            .for_each(|(idx, el)| {
                *el = self.wavelet_transform_bottom_half_element(top_half, idx);
            });

        Ok(())
    }
}

impl<T: Float> WaveletTransformFloating<T> for Matrix2d<T> {
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
                    low_pass_filter[(hi + hi_end) as usize] * self.data[0][x_position as usize];
            }
            result.data[0][i] = sum_value_low;

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
                    high_pass_filter[(gi + gi_end) as usize] * self.data[0][x_position as usize];
            }
            result.data[0][i + size] = sum_value_high;
        }
        Ok(())
    }
}

impl<T: Number> FirFilter<T> for Matrix2d<T> {
    fn fir_filter_element(&self, kernel: &Self, element_idx: usize) -> T {
        let mut sum = T::zero();
        for j in 0..kernel.cols {
            let idx = element_idx as isize + (j as isize - kernel.cols as isize + 1);
            if idx >= 0 && idx < self.cols as isize {
                sum += kernel.data[0][kernel.cols - j - 1] * self.data[0][idx as usize];
            }
        }
        sum
    }

    /// Does vector convolution using the kernel passed  
    /// kernel.cols needs to be odd and result.cols = self.cols + kernel.cols - 1
    fn fir_filter(&self, kernel: &Self, result: &mut Self) -> Result<(), Error> {
        if kernel.rows != 1
            || self.rows != 1
            || result.rows != 1
            || result.cols != self.cols + kernel.cols - 1
        {
            return Err(Error::InvalidDimensions);
        }
        result.data[0]
            .iter_mut()
            .enumerate()
            .for_each(|(idx, result_el)| {
                *result_el = self.fir_filter_element(kernel, idx);
            });
        Ok(())
    }
}

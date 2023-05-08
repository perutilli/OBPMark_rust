use crate::matrix_1d::Matrix1d;
use crate::number_traits::Number;
use crate::rayon_traits::*;
use crate::Error;

use rayon::prelude::*;

impl<T: Number> RayonMatMul for Matrix1d<T> {
    fn rayon_multiply(&self, other: &Self, result: &mut Self) -> Result<(), Error> {
        if self.cols != other.rows || self.rows != result.rows || other.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result
            .data
            .par_chunks_mut(other.cols)
            .enumerate()
            .for_each(|(i, chunk)| {
                for (j, el) in chunk.iter_mut().enumerate() {
                    let mut sum = T::zero();
                    for k in 0..self.cols {
                        sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                    }
                    *el = sum;
                }
            });

        Ok(())
    }
}

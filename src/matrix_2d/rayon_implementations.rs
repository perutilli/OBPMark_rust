use crate::matrix_2d::Matrix2d;
use crate::number_traits::Number;
use crate::rayon_traits::*;
use crate::Error;

use rayon::prelude::*;

impl<T: Number> RayonMatMul for Matrix2d<T> {
    fn rayon_multiply(&self, other: &Self, result: &mut Self) -> Result<(), Error> {
        if self.cols != other.rows || self.rows != result.rows || other.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        result.data.par_iter_mut().enumerate().for_each(|(i, row)| {
            for (j, el) in row.iter_mut().enumerate() {
                let mut sum = T::zero();
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                *el = sum;
            }
        });

        Ok(())
    }
}

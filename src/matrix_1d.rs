use std::fmt::Display;

use crate::{format_number, BaseMatrix, Error, MatMul, Number, Relu, Softmax};

pub struct Matrix {
    data: Vec<Number>,
    rows: usize,
    cols: usize,
}

impl BaseMatrix for Matrix {
    fn new(data: Vec<Vec<Number>>, rows: usize, cols: usize) -> Matrix {
        Matrix {
            data: data.into_iter().flatten().collect(),
            rows,
            cols,
        }
    }

    fn get_data(&self) -> Vec<Vec<Number>> {
        self.data
            .clone()
            .chunks(self.cols)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<Number>>>()
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.data
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    if i % self.cols != self.cols - 1 {
                        // if we are not at the end of the row
                        format!("{} ", format_number(x))
                    } else {
                        // if we are at the end of the row
                        format!("{}\n", format_number(x))
                    }
                })
                .collect::<Vec<String>>()
                .join(""),
        )
    }
}

impl MatMul for Matrix {
    fn multiply(&self, other: &Matrix, result: &mut Matrix) -> Result<(), Error> {
        if self.cols != other.rows {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = Number::default();
                // NOTE: this allows result to not be all zeros
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * self.rows + j] = sum;
            }
        }
        Ok(())
    }
}

impl Relu for Matrix {
    fn relu(&self, result: &mut Matrix) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i * self.rows + j] =
                    self.data[i * self.rows + j].max(Number::default());
            }
        }
        Ok(())
    }
}

#[cfg(not(feature = "int"))]
impl Softmax for Matrix {
    fn softmax(&self, result: &mut Matrix) -> Result<(), Error> {
        if self.rows != result.rows || self.cols != result.cols {
            return Err(Error::InvalidDimensions);
        }
        for i in 0..self.rows {
            let mut sum = Number::default();
            for j in 0..self.cols {
                let val = self.data[i * self.rows + j].exp();
                sum += val;
                result.data[i * self.rows + j] = val;
            }
            for j in 0..self.cols {
                result.data[i * self.rows + j] /= sum;
            }
        }
        Ok(())
    }
}

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[cfg(feature = "float")]
pub type Number = f32;
#[cfg(feature = "double")]
pub type Number = f64;
#[cfg(feature = "int")]
pub type Number = i32;
#[cfg(not(any(feature = "float", feature = "double", feature = "int")))]
pub type Number = f32;

#[derive(Debug)]
pub enum Error {
    InvalidDimensions,
}

pub trait MatMul {
    fn multiply(&self, other: &Self, result: &mut Self) -> Result<(), Error>;
}

pub fn random_matrix(seed: u64, rows: usize, cols: usize) -> Vec<Vec<Number>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![vec![Number::default(); cols]; rows];
    for row in &mut data {
        for col in row {
            *col = rng.gen();
        }
    }
    data
}

// TODO: we might want to change this, ok for now
// TODO: right now we will overflow for ints, need to fix that
pub trait FromRandomSeed {
    fn from_random_seed(seed: u64, rows: usize, cols: usize) -> Self;
}

pub mod matrix_1d;
pub mod matrix_2d;
pub mod matrix_ndarray;

/*
pub mod matrix {

    #![allow(non_snake_case)] // TODO: decide if we want to keep this or not
    use rand::Rng;
    use std::fmt::Display;
    use std::fmt::{self, Formatter};
    use std::ops::Mul;

    #[cfg(feature = "float")]
    pub type Number = f32;
    #[cfg(feature = "double")]
    pub type Number = f64;
    #[cfg(feature = "int")]
    pub type Number = i32;
    #[cfg(not(any(feature = "float", feature = "double", feature = "int")))]
    pub type Number = f32;
    // default to float, this solution is more than anything for linting reasons
    // TODO: find a better way to do this

    // TODO: move the functions inside the Matrix struct as helper methods

    pub fn matrix_multiplication(
        A: &Vec<Vec<Number>>,
        B: &Vec<Vec<Number>>,
        C: &mut Vec<Vec<Number>>,
    ) -> () {
        let n = A.len();
        let m = A[0].len();
        let w = B[0].len();
        assert_eq!(
            m,
            B.len(),
            "Matrix dimensions do not match, impossible to multiply"
        );

        // let mut C = vec![vec![Number::default(); w]; n];
        for i in 0..n {
            for j in 0..w {
                for k in 0..m {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    pub fn matrix_multiplication_1d(
        A: &Vec<Number>,
        B: &Vec<Number>,
        C: &mut Vec<Number>,
        n: usize,
        m: usize,
        w: usize,
    ) -> () {
        for i in 0..n {
            for j in 0..w {
                for k in 0..m {
                    C[i * n + j] += A[i * n + k] * B[k * w + j];
                }
            }
        }
    }

    pub fn matrix_multiplication_ndarray(
        A: &ndarray::Array2<Number>,
        B: &ndarray::Array2<Number>,
        C: &mut ndarray::Array2<Number>,
    ) -> () {
        *C = A.dot(B);
    }

    /// Generates a random square matrix with side length `size`
    pub fn generate_random_matrix(size: usize) -> Vec<Vec<Number>> {
        let mut rng = rand::thread_rng();
        let mut matrix = vec![vec![Number::default(); size]; size];
        for i in 0..size {
            for j in 0..size {
                #[cfg(feature = "int")] // this is necessary to avoid compilation errors
                {
                    // this is here to avoid overflow when multiplying
                    matrix[i][j] = rng.gen_range(0..100);
                }
                if !cfg!(feature = "int") {
                    matrix[i][j] = rng.gen::<Number>();
                }
            }
        }
        matrix
    }

    /*
    /// Converts a matrix to a string
    pub fn matrix_to_str(matrix: Vec<Vec<Number>>) -> String {
        let mut string = String::new();
        for row in matrix {
            for element in row {
                // note that the :.5 does not have any effect on integers
                string.push_str(&format!("{:.5} ", element));
            }
            string.push_str("\n");
        }
        string
    }
     */

    enum Data {
        Vec2d(Vec<Vec<Number>>),
        Vec1d(Vec<Number>),
        NdArray(ndarray::Array2<Number>),
    }

    pub enum MatType {
        Vec2d,
        Vec1d,
        NdArray,
    }

    #[derive(Debug)]
    pub enum Error {
        IncompatibleMatrixTypes,
        InvalidDimensions,
    }

    pub struct Matrix {
        data: Data,
        side_a: usize,
        side_b: usize,
    }

    impl Matrix {
        pub fn new(
            matrix: Vec<Vec<Number>>,
            side_a: usize,
            side_b: usize,
            mat_type: MatType,
        ) -> Matrix {
            match mat_type {
                MatType::Vec2d => Matrix {
                    data: Data::Vec2d(matrix),
                    side_a,
                    side_b,
                },
                MatType::Vec1d => Matrix {
                    data: Data::Vec1d(matrix.into_iter().flatten().collect()),
                    side_a,
                    side_b,
                },
                MatType::NdArray => Matrix {
                    data: Data::NdArray(
                        ndarray::Array2::from_shape_vec(
                            (side_a, side_b),
                            matrix.into_iter().flatten().collect(),
                        )
                        .unwrap(),
                    ),
                    side_a,
                    side_b,
                },
            }
        }

        pub fn multiply(&self, other: &Matrix, result: &mut Matrix) -> Result<(), Error> {
            // Here we would want to change 2 things:
            // 1. Make this function overload the * operator
            // 2. Have the subcalls be methods of Matrix instead of functions
            //    (and make them return a Result<(), Error>)
            match (&self.data, &other.data, &mut result.data) {
                (Data::Vec2d(a), Data::Vec2d(b), Data::Vec2d(c)) => {
                    matrix_multiplication(a, b, c);
                    Ok(())
                }
                (Data::Vec1d(a), Data::Vec1d(b), Data::Vec1d(c)) => {
                    matrix_multiplication_1d(a, b, c, self.side_a, self.side_b, other.side_b);
                    Ok(())
                }
                (Data::NdArray(a), Data::NdArray(b), Data::NdArray(c)) => {
                    matrix_multiplication_ndarray(a, b, c);
                    Ok(())
                }
                _ => Err(Error::IncompatibleMatrixTypes),
            }
        }
    }

    impl Display for Matrix {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
            match &self.data {
                Data::Vec2d(data) => write!(
                    f,
                    "{}",
                    data.iter()
                        .map(|x| x
                            .iter()
                            .map(|y| format!("{:.6}", y))
                            .collect::<Vec<String>>()
                            .join(" "))
                        .collect::<Vec<String>>()
                        .join("\n")
                ),

                Data::Vec1d(data) => write!(
                    f,
                    "{}",
                    data.iter()
                        .enumerate()
                        .map(|(i, x)| {
                            if i % self.side_b != self.side_b - 1 {
                                // if we are not at the end of the row
                                format!("{:.6} ", x)
                            } else {
                                // if we are at the end of the row
                                format!("{:.6}\n", x)
                            }
                        })
                        .collect::<Vec<String>>()
                        .join("")
                ),

                Data::NdArray(data) => write!(
                    f,
                    "{}",
                    data.axis_iter(ndarray::Axis(0))
                        .map(|x| x
                            .iter()
                            .map(|y| format!("{:.6}", y))
                            .collect::<Vec<String>>()
                            .join(" "))
                        .collect::<Vec<String>>()
                        .join("\n")
                ),
            }
            // have this call helper methods to convert the different types of data into a string
        }
    }

    // NOTE: this moves the allocation inside the call, so we probably don't want to do this when benchmarking
    // Also this creates the possibility of a panic, while using .multiply() does not
    impl Mul<Matrix> for Matrix {
        type Output = Matrix;

        fn mul(self, other: Matrix) -> Matrix {
            let mat_type = match (&self.data, &other.data) {
                (Data::Vec2d(_), Data::Vec2d(_)) => MatType::Vec2d,
                (Data::Vec1d(_), Data::Vec1d(_)) => MatType::Vec1d,
                (Data::NdArray(_), Data::NdArray(_)) => MatType::NdArray,
                _ => panic!("Incompatible matrix types"),
            };
            let mut result = Matrix::new(
                vec![vec![Number::default(); self.side_b]; other.side_a],
                self.side_a,
                other.side_b,
                mat_type,
            );
            self.multiply(&other, &mut result).unwrap();
            result
        }
    }
}
*/

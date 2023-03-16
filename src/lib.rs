pub mod matrix {

    #![allow(non_snake_case)] // TODO: decide if we want to keep this or not
    use rand::Rng;

    #[cfg(feature = "float")]
    pub type Number = f32;
    #[cfg(feature = "double")]
    pub type Number = f64;
    #[cfg(feature = "int")]
    pub type Number = i32;
    #[cfg(not(any(feature = "float", feature = "double", feature = "int")))]
    pub type Number = f32; // default to float

    // TODO: check with Leonidas if he has any idea why this is so slow
    // probably will end up using 1d arrays, maybe also ndarray
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

    /*  Here for testing the overhead of Vec
        The time for 100x100 matrices is about 15-20 ms
        compared to 1 ms for the C sequential implementation
    pub fn matrix_multiplication_array(
        A: &[Number; 100 * 100],
        B: &[Number; 100 * 100],
        C: &mut [Number; 100 * 100],
    ) -> () {
        for i in 0..100 {
            for j in 0..100 {
                for k in 0..100 {
                    C[i * 100 + j] += A[i * 100 + k] * B[k * 100 + j];
                }
            }
        }
    }
    */

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
}

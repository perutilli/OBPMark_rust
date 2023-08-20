#![no_std]

#[cfg(feature = "float")]
type CType = core::ffi::c_float;
#[cfg(feature = "double")]
type CType = core::ffi::c_double;
#[cfg(feature = "int")]
type CType = core::ffi::c_int;
#[cfg(feature = "half")]
compile_error!("Half precision validation not supported yet");
#[cfg(not(any(
    feature = "float",
    feature = "double",
    feature = "int",
    feature = "half"
)))]
type CType = core::ffi::c_float;

extern "C" {
    // void matrix_multiplication(const bench_t *A, const bench_t *B, bench_t *C, const unsigned int n, const unsigned int m, const unsigned int w)
    pub fn matrix_multiplication(
        a: *const CType,
        b: *const CType,
        c: *mut CType,
        n: usize,
        m: usize,
        k: usize,
    );

    // void relu(const bench_t *A, bench_t *B, const unsigned int size)void relu(const bench_t *A, bench_t *B, const unsigned int size)
    pub fn relu(a: *const CType, b: *mut CType, size: usize);

    // void softmax(const bench_t *A, bench_t *B, const unsigned int size)
    pub fn softmax(a: *const CType, b: *mut CType, size: usize);

    // void matrix_convolution(const bench_t *A, bench_t *kernel, bench_t *B, const int size, const int kernel_size)
    pub fn matrix_convolution(
        a: *const CType,
        kernel: *const CType,
        b: *mut CType,
        size: usize,
        kernel_size: usize,
    );

    // void max_pooling(const bench_t* A, bench_t* B,const unsigned int size,const unsigned int stride,  const unsigned int lateral_stride)
    pub fn max_pooling(
        a: *const CType,
        b: *mut CType,
        size: usize,
        stride: usize,
        lateral_stride: usize, // lateral_stride should be size/stride
    );

    /// Note that half_size is half of the size of the output array.
    // void ccsds_wavelet_transform(const bench_t* A, bench_t* B, const int size)
    pub fn ccsds_wavelet_transform(a: *const CType, b: *mut CType, half_size: usize);

    // void vector_convolution(const bench_t* A, bench_t* kernel, bench_t* B,const int size, const int kernel_size)
    pub fn vector_convolution(
        a: *const CType,
        kernel: *const CType,
        b: *mut CType,
        size: usize,
        kernel_size: usize,
    );

    // need to fix my code before I can use this
    // void correlation_2D(const bench_t *A, const bench_t *B, result_bench_t *R, const int size)
    // pub fn correlation_2d(a: *const CType, b: *const CType, r: *mut CType, size: usize);
}

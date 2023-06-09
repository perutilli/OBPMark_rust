use crate::number_traits::Float;
use crate::{Error, Padding};

pub trait ParallelMatMul {
    fn parallel_multiply(
        &self,
        other: &Self,
        result: &mut Self,
        n_threads: usize,
    ) -> Result<(), Error>;
}

pub trait ParallelConvolution {
    fn parallel_convolute(
        &self,
        kernel: &Self,
        padding: Padding,
        result: &mut Self,
        n_threads: usize,
    ) -> Result<(), Error>;
}

pub trait ParallelRelu {
    fn parallel_relu(&self, result: &mut Self, n_threads: usize) -> Result<(), Error>;
}

pub trait ParallelSoftmax {
    fn parallel_softmax(&self, result: &mut Self, n_threads: usize) -> Result<(), Error>;
}

pub trait ParallelMaxPooling {
    fn parallel_max_pooling(
        &self,
        result: &mut Self,
        row_stride: usize,
        col_stride: usize,
        n_threads: usize,
    ) -> Result<(), Error>;
}

pub trait ParallelLRN<T: Float> {
    fn parallel_lrn(
        &self,
        result: &mut Self,
        alpha: T,
        beta: T,
        k: T,
        n_threads: usize,
    ) -> Result<(), Error>;
}

pub trait ParallelFiniteImpulseResponseFilter {
    fn parallel_fir_filter(
        &self,
        kernel: &Self,
        result: &mut Self,
        n_threads: usize,
    ) -> Result<(), Error>;
}

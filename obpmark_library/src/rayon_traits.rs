use crate::number_traits::Float;
use crate::{Error, FastFourierTransformHelper, Padding};

pub trait RayonMatMul {
    fn rayon_multiply(&self, other: &Self, result: &mut Self) -> Result<(), Error>;
}

pub trait RayonMaxPooling {
    fn rayon_max_pooling(
        &self,
        result: &mut Self,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<(), Error>;
}

pub trait RayonSoftmax {
    fn rayon_softmax(&self, result: &mut Self) -> Result<(), Error>;
}

pub trait RayonConvolution {
    fn rayon_convolute(
        &self,
        kernel: &Self,
        padding: Padding,
        result: &mut Self,
    ) -> Result<(), Error>;
}

pub trait RayonRelu {
    fn rayon_relu(&self, result: &mut Self) -> Result<(), Error>;
}

pub trait RayonLRN<T: Float> {
    fn rayon_lrn(&self, result: &mut Self, alpha: T, beta: T, k: T) -> Result<(), Error>;
}

pub trait RayonFiniteImpulseResponseFilter {
    fn rayon_fir_filter(&self, kernel: &Self, result: &mut Self) -> Result<(), Error>;
}

pub trait RayonFastFourierTransformWindowed<T>: FastFourierTransformHelper<T> {
    fn rayon_fft_windowed(&self, window: usize, result: &mut Self) -> Result<(), Error>;
}

pub trait RayonCorrelation {
    type Output;
    fn rayon_correlate(&self, other: &Self) -> Result<Self::Output, Error>;
}

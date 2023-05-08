use crate::{Error, Padding};

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

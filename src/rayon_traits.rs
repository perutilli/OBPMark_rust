use crate::Error;

pub trait RayonMatMul {
    fn rayon_multiply(&self, other: &Self, result: &mut Self) -> Result<(), Error>;
}

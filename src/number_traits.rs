use half::f16;
use rand::rngs::StdRng;
use rand::Rng;

pub trait Fundamental:
    'static
    + Sized
    + Send
    + Sync
    + Unpin
    + Clone
    + Copy
    + Default
    + std::str::FromStr
    + PartialEq<Self>
    + PartialOrd<Self>
    + std::fmt::Debug
    + std::fmt::Display
{
} // from funty

impl Fundamental for f32 {}
impl Fundamental for f64 {}
impl Fundamental for i32 {}
impl Fundamental for f16 {}

pub trait Serialize {
    type Bytes;
    fn to_be_bytes(self) -> Self::Bytes;
    fn to_le_bytes(self) -> Self::Bytes;
    fn to_ne_bytes(self) -> Self::Bytes;
    fn from_be_bytes(bytes: Self::Bytes) -> Self;
    fn from_le_bytes(bytes: Self::Bytes) -> Self;
    fn from_ne_bytes(bytes: Self::Bytes) -> Self;
}

macro_rules! impl_serialize {
    ($type: ty) => {
        impl Serialize for $type {
            type Bytes = [u8; core::mem::size_of::<$type>()];
            fn to_be_bytes(self) -> Self::Bytes {
                self.to_be_bytes()
            }
            fn to_le_bytes(self) -> Self::Bytes {
                self.to_le_bytes()
            }
            fn to_ne_bytes(self) -> Self::Bytes {
                self.to_ne_bytes()
            }
            fn from_be_bytes(bytes: Self::Bytes) -> Self {
                <$type>::from_be_bytes(bytes)
            }
            fn from_le_bytes(bytes: Self::Bytes) -> Self {
                <$type>::from_le_bytes(bytes)
            }
            fn from_ne_bytes(bytes: Self::Bytes) -> Self {
                <$type>::from_ne_bytes(bytes)
            }
        }
    };
}

impl_serialize!(f32);
impl_serialize!(f64);
impl_serialize!(i32);
impl_serialize!(f16);

pub trait RngRange {
    fn gen_range(rng: &mut StdRng, min: Self, max: Self) -> Self;
}

// requires the type to implement SampleUniform
macro_rules! impl_rng_range {
    ($t: ty) => {
        impl RngRange for $t {
            fn gen_range(rng: &mut StdRng, min: Self, max: Self) -> Self {
                rng.gen_range(min..max)
            }
        }
    };
}

impl_rng_range!(f32);
impl_rng_range!(f64);
impl_rng_range!(i32);

impl RngRange for f16 {
    fn gen_range(rng: &mut StdRng, min: Self, max: Self) -> Self {
        let min = f32::from(min);
        let max = f32::from(max);
        // NOTE: this should be ok as min and max are f16, hence the
        // conversion should not give +inf or -inf
        f16::from_f32(rng.gen_range(min..max))
    }
}

pub trait Number:
    num_traits::NumAssignRef
    + RngRange
    + Serialize
    + Fundamental
    + RngRange
    + num_traits::NumRef
    + for<'a> std::iter::Sum<&'a Self>
    + std::iter::Sum<Self>
    + num_traits::AsPrimitive<f64>
{
}

impl Number for f32 {}
impl Number for f64 {}
impl Number for i32 {}
impl Number for f16 {}

pub trait Float: Number + num_traits::Float {}

impl Float for f32 {}
impl Float for f64 {}
impl Float for f16 {}

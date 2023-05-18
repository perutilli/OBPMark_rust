use crate::Error;

pub trait ParallelMatMul {
    fn parallel_multiply(
        &self,
        other: &Self,
        result: &mut Self,
        n_threads: usize,
    ) -> Result<(), Error>;
}

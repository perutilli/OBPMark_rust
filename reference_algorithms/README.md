For use with RiscV embedded target, you need to disable the crate default features and set the correct compiler to be used. This is done using an enviroment variable when compiling your crate that depends on this crate, for example:
```bash
export CC=riscv64-unknown-elf-gcc # assuming it is in the PATH, otherwise use the full path
```
To allow compilation for different targets, you can specify the compiler for a specific architecture:
```bash
export CC_riscv64gc_unknown_none_elf=riscv64-unknown-elf-gcc
```
The name of the variable for different targets is CC_{target with - replaced with _}. The target is the rustup toolchain used.
Right now only std and riscv64gc-unknown-none-elf targets are tested to work. 32 bit RiscV targets will most likely not work, due to specified floating point ABI with support for 64 bit floating points. If you need to add support, you should edit the `build.rs` file, in particular pay attention to the `.flag("-mabi=lp64d")`, which should be changed to `-mabi=ilp32f` for 32 bit targets with hard float support. More infomation about gcc ABI flags for RiscV can be found [here](https://gcc.gnu.org/onlinedocs/gcc/RISC-V-Options.html).
## Important usage note
As cargo features are supposed to be additive (i.e. not mutually exclusive), enforcing that features `riscv_hard_float` and `std` are not enabled at the same time will result in rust-analyzer complaining that a compilation error will happen if we compile the whole crate. To avoid this we made `std` a preferential feature, so that if both are enabled, `std` will be used. This does not let you compile the whole crate, but it removes the annoying error from rust-analyzer.
As specified in the workspace README, it in NOT intended to be compiled as one, but rather you should compile the single needed crate (i.e. benchmaks or bare_metal) depending on your target.
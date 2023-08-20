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
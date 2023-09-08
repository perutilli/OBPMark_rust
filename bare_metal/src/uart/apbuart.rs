use core::fmt::Error;
use core::fmt::Write;

pub struct Uart {
    base_address: usize,
    status_address: usize,
    _control_address: usize,
}

impl Write for Uart {
    fn write_str(&mut self, out: &str) -> Result<(), Error> {
        for c in out.bytes() {
            self.put(c);
        }
        Ok(())
    }
}

impl Uart {
    pub fn new(base_address: usize) -> Self {
        Uart {
            base_address,
            status_address: base_address + 0x4,
            _control_address: base_address + 0x8,
        }
    }

    pub fn init(&mut self) {
        // let ptr = self.base_address as *mut u32;

        // METASAT uart

        // 0x8 control register
        // bit 31 -> FIFO available, read only
        // bit 10 -> reciever fifo interrupt enable -> for now not needed
        // bit 1 -> transmitter enable
        // bit 0 -> reciever enable
        // ptr.add(0x9).write_volatile(0x0);
        // ptr.add(0x8).write_volatile(0b11);

        // 100 MHz 38343 BAUD
        // 0xC scaler register
        // scaler value =  (system_clock_frequency) / (baud_rate * 8) - 1
        // let scaler_value: u32 = 100_000_000 / (38_343 * 8) - 1;
        // let lowest_byte: u8 = (scaler_value & 0xff).try_into().unwrap();
        // let top_byte: u8 = ((scaler_value & 0xff00) >> 8).try_into().unwrap();
        // ptr.add(0xC).write_volatile(lowest_byte);
        // ptr.add(0xC).add(1).write_volatile(top_byte);

        // ptr.add(0x8).write_volatile(0x8000_0003);
        // ptr.add(0xC).write_volatile(0x0000_0145);
        // // this is the value written by forward enable
        // // which is the same as the result of the formula

        write!(self, "UART initialized\r\n").unwrap();
    }

    pub fn put(&mut self, c: u8) {
        let ptr = self.base_address as *mut u8;
        unsafe {
            // wait while transmitter queue full
            while (self.status_address as *mut u32).read_volatile() & 0x200 != 0 {}
            ptr.write_volatile(c);
        }
    }
}

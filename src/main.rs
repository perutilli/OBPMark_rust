fn main() {
    let mut frame = gen_random_frame16(5, 5, u16::max_value());
    let offsets = gen_random_frame16(5, 5, 100);
    println!("{:?}", frame);
    println!("{:?}", offsets);
    frame.f_offset(&offsets);
    println!("{:?}", frame);
    let gains = gen_random_frame16(5, 5, u16::max_value());
    println!("{:?}", gains);
    frame.f_gain(&gains);
    println!("{:?}", frame);
    let bad_pixels = Mask::new(
        vec![
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, true, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
        ],
        5,
        5,
    );
    println!("{:?}", bad_pixels);
    frame.f_mask_replace(&bad_pixels);
    println!("{:?}", frame);
}

#[derive(Debug)]
struct Mask {
    frame: Vec<Vec<bool>>,
    width: usize,
    height: usize,
}

impl Mask {
    fn new(frame: Vec<Vec<bool>>, width: usize, height: usize) -> Mask {
        Mask {
            frame,
            width,
            height,
        }
    }
}

#[derive(Debug)]
struct Frame16 {
    frame: Vec<Vec<u16>>,
    width: usize,
    height: usize,
}

#[derive(Debug)]
struct Frame32 {
    frame: Vec<Vec<u32>>,
    width: usize,
    height: usize,
}

impl Frame16 {
    fn new(frame: Vec<Vec<u16>>, width: usize, height: usize) -> Frame16 {
        Frame16 {
            frame,
            width,
            height,
        }
    }

    fn f_offset(&mut self, offsets: &Self) {
        for x in 0..self.width {
            for y in 0..self.height {
                assert!(self.frame[x][y] >= offsets.frame[x][y]);
                self.frame[x][y] -= offsets.frame[x][y];
            }
        }
    }

    fn f_gain(&mut self, gains: &Self) {
        for x in 0..self.width {
            for y in 0..self.height {
                self.frame[x][y] =
                    (self.frame[x][y] as u32 * gains.frame[x][y] as u32 >> 16) as u16;
            }
        }
    }

    fn f_neighbour_masked_sum(&self, bad_pixels: &Mask, x_mid: usize, y_mid: usize) -> u16 {
        /* Calculate unweighted sum of good pixels in 3x3 neighbourhood (can be smaller if on edge or corner) of (x_mid, y_mid) */
        let mut sum = 0;
        let mut n_sum = 0;

        assert!(
            self.width == bad_pixels.width && self.height == bad_pixels.height,
            "Mask and frame must have same dimensions"
        );

        /* TODO: 3 should be a parameter or constant */

        for dx in -1..2 {
            for dy in -1..2 {
                let x = x_mid as i32 + dx;
                let y = y_mid as i32 + dy;
                if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
                    if !bad_pixels.frame[x as usize][y as usize] {
                        sum += self.frame[x as usize][y as usize] as u32;
                        n_sum += 1;
                    }
                }
            }
        }

        (sum / n_sum) as u16
    }

    fn f_mask_replace(&mut self, bad_pixels: &Mask) {
        /* replace bad pixels with mean of neighbours calculated by f_neighbour_masked_sum */
        for x in 0..self.width {
            for y in 0..self.height {
                if bad_pixels.frame[x][y] {
                    self.frame[x][y] = self.f_neighbour_masked_sum(bad_pixels, x, y);
                }
            }
        }
    }

    /// Substitutes pixels above threshold with mean of temporal neighbours
    /// threshold = 2 * mean of (4) temporal neighbours
    fn f_scrub(&mut self, fs: &Vec<Self>, frame_i: usize) {
        for x in 0..self.width {
            for y in 0..self.height {
                let sum = fs[frame_i - 2].frame[x][y] as u32
                    + fs[frame_i - 1].frame[x][y] as u32
                    + fs[frame_i + 1].frame[x][y] as u32
                    + fs[frame_i + 2].frame[x][y] as u32;
                let mean = sum / 4;
                let thr = 2 * mean;
                if self.frame[x][y] as u32 > thr {
                    self.frame[x][y] = mean as u16;
                }
            }
        }
    }

    fn f_2x2_bin(&self) -> Frame32 {
        let mut binned_frame = Frame32::new(
            vec![vec![0; self.height / 2]; self.width / 2],
            self.width / 2,
            self.height / 2,
        );
        for x in 0..self.width {
            for y in 0..self.height {
                binned_frame.frame[x][y] = self.frame[x][y] as u32
                    + self.frame[x + 1][y] as u32
                    + self.frame[x][y + 1] as u32
                    + self.frame[x + 1][y + 1] as u32;
            }
        }
        binned_frame
    }
}

impl Frame32 {
    fn new(frame: Vec<Vec<u32>>, width: usize, height: usize) -> Frame32 {
        Frame32 {
            frame,
            width,
            height,
        }
    }

    fn f_coadd(&mut self, add_frame: &Self) {
        for x in 0..self.width {
            for y in 0..self.height {
                self.frame[x][y] += add_frame.frame[x][y];
            }
        }
    }
}

fn gen_random_frame16(width: usize, height: usize, max: u16) -> Frame16 {
    let mut frame = vec![vec![0; height]; width];
    for x in 0..width {
        for y in 0..height {
            frame[x][y] = rand::random::<u16>() % max;
        }
    }
    Frame16::new(frame, width, height)
}

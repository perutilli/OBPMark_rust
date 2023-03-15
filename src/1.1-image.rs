
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
                    ((self.frame[x][y] as u32 * gains.frame[x][y] as u32) >> 16) as u16;
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

    /* MOVED TO ImageData because it uses collection of frames from it
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
    */

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

pub struct ImageData {
    frames: Vec<Frame16>,
    num_frames: usize,
    offsets: Frame16,
    gains: Frame16,
    bad_pixels: Mask,

    scrub_mask: Mask,

    binned_frame: Frame32,

    image_output: Frame32,
}

impl ImageData {
    fn f_scrub(&mut self, frame_idx: usize) {
        for x in 0..self.frames[frame_idx].width {
            for y in 0..self.frames[frame_idx].height {
                let sum = self.frames[frame_idx - 2].frame[x][y] as u32
                    + self.frames[frame_idx - 1].frame[x][y] as u32
                    + self.frames[frame_idx + 1].frame[x][y] as u32
                    + self.frames[frame_idx + 2].frame[x][y] as u32;
                let mean = sum / 4;
                let thr = 2 * mean;
                if self.frames[frame_idx].frame[x][y] as u32 > thr {
                    self.frames[frame_idx].frame[x][y] = mean as u16;
                }
            }
        }
    }

    fn process_image(&mut self) {
        for i in 0..self.num_frames {
            self.prepare_image_frame(i);
            self.proc_image_frame(i);
        }
    }

    fn prepare_image_frame(&mut self, frame_idx: usize) {
        self.frames[frame_idx].f_offset(&self.offsets);
        self.frames[frame_idx].f_mask_replace(&self.bad_pixels);
    }

    fn proc_image_frame(&mut self, frame_idx: usize) {
        self.f_scrub(frame_idx);
        self.frames[frame_idx].f_gain(&self.gains);
        self.binned_frame = self.frames[frame_idx].f_2x2_bin();
        self.image_output.f_coadd(&self.binned_frame);
    }
}

mod from_files {

    // the funcion should receive a path to a folder with files
    // and fill an image data structure with the data
    // there is one file for each frame (check the naming convention)
    // and one file each for offsets, gains and bad pixels
    // plus 4 files for the scrub masks at t-2, t-1, t+1, t+2 (TODO: understand what they are for)
    // scrub function does not use any mask to my knowledge
}

#[cfg(test)]
mod kernel_tests {
    #[test]
    fn offset_test() {
        let mut frame =
            super::Frame16::new(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]], 3, 3);
        let offsets = super::Frame16::new(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]], 3, 3);
        frame.f_offset(&offsets);
        assert_eq!(
            frame.frame,
            vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]]
        );
    }

    #[test]
    fn gain_test() {
        let mut frame = super::Frame16::new(vec![vec![2, 4, 6]; 3], 3, 3);
        let gains = super::Frame16::new(
            vec![vec![u16::max_value(), u16::max_value(), u16::max_value()]; 3],
            3,
            3,
        );
        frame.f_gain(&gains);
        assert_eq!(frame.frame, vec![vec![1, 3, 5]; 3]);
    }
}

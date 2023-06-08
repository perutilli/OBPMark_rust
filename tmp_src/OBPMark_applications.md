# 09/03 Call with Leonidas
- Call the c seq as a validation step
- We might need to port the board support package for the device
- Maybe raspberry pi to start with (is it multicore?), we also have other devices at bsc
# Main questions
- Are the benchmarks set up to run on different devices? The init function does not seem to do anything, and platform and device are statically set to 0 as device_name is set to "Generic device". 
- Are the frames stored by columns in the binary files? It looks like it, from the code. Also is the duplicated code in image_file_util.c there for compatibility with the original code? Or could the functions be merged?
- Should I keep the support for both 1d and 2d buffers or can I only use 2d buffers?
- Questions in the processing functions section, about the reasons to carry out the operations in the processing pipeline
- `number_neighbours`, both for the radiation scrubbing (temporal) and for the spatial case is statically set to 4, why? Should this be a parameter or in a config?
- Should we allow wrapping in the offset function? This does not seem right, but the random generator for the offset would lead to this behaviour I'm quite sure
- How does the device running work? It does not seem like it is running on another device, but then why all the memcpys etc?
# Action plan
- [x] Port the frame structures
- [x] Port the processing functions for the 1.1-image benchmark
  
# General notes

Hip should be to run on gpus both nvidia and not.

# 1.1-image 

## What is the benchmark trying to do

## Function main

Initialize the arguments with `arguments_handler`, this are the arguments:
```
bool csv_mode = false;
bool print_output = false;
bool verbose_output = false;
bool database_mode = false;
bool random_data = false;
bool no_output_file = false;
bool extended_csv_mode = false;

int file_loading_output = 0;

unsigned int w_size = 0;
unsigned int h_size = 0; 
unsigned int num_frames = 0; 
unsigned int num_processing_frames = 0;
```
Calculate the following values from the arguments:
```
unsigned int size_frame;
unsigned int mem_size_frame;
unsigned int mem_size_bad_map;
unsigned int size_reduction_image;
unsigned int mem_size_reduction_image;
```
### `number_neighbours` is statically set to 4, why?  
(From here on we'll assume that the frames are a 1d buffer, depending on the value of flag `OBPMARK_FRAME_DATA_2D` they could also be stored as 2d buffers, but actually the cpp code at the moment is not able to deal with the 2d buffers)  
`input_frames` allocated to store the input images of type `frame16_t`:
```
typedef struct {
    uint16_t *f;   ///< Frame buffer
    unsigned int w; ///< Frame width
    unsigned int h; ///< Frame height
} frame16_t;
```
### Should the names of the fields stay the same or can I use more meaningful names?
`output_image` allocated to store the output image of type `frame32_t`
```
typedef struct {
    uint32_t *f;   ///< Frame buffer
    unsigned int w; ///< Frame width
    unsigned int h; ///< Frame height
} frame32_t;
```
Three variable are allocated (`offset_map`, `bad_pixel_map` and `gain_map`, respectively of types `frame16_t`, `frame8_t` and `frame16_t`) along with their buffers.  
`input_frames, output_image, offset_map, bad_pixel_map, gain_map, w_size, h_size, num_frames` are either read from a file or randomly generated, when reading from file there can be an error in which case the benchmark is aborted.  
`init_benchmark` function is called, this actually runs the benchmark.  
All the memory allocated is freed (not necessary in rust)

## Function init_benchmark

Recieves the following parameters:
```
typedef struct {
frame16_t *input_frames,
frame32_t *output_image,

frame16_t *offset_map,
frame8_t *bad_pixel_map, 
frame16_t *gain_map,

unsigned int w_size,
unsigned int h_size,
unsigned int num_frames,

long int timestamp,
print_info_data_t *benchmark_info
```
It allocates structure image_data of type `image_data_t`
```
struct image_data_t
{
	frame16_t *frames;
	unsigned int num_frames; 

	frame16_t offsets;
	frame16_t gains; 
	frame8_t bad_pixels;

	frame8_t scrub_mask;

	frame32_t binned_frame; 

	frame32_t image_output; 
};
```
Bunch of operations that have to do with the device such as `init` and `device_memory_init`.  
Call function `process_benchmark`, that runs the benchmark on the device.  
Copies the result back to the host (`copy_memory_to_host`), gets the time (`get_elapsed_time`) and prints the results.

## Function `process_benchmark` (`ports/cpu/device.c`)

## Processing functions

- `f_offset`: bias offset correction, each pixel value in the frame is reduced by the offset value at the same position (why do we do this? different brightnesses due to sensor uneveness?)
- `f_mask_replace`: bad pixel correction, if the pixel is a bad pixel, it is replaced with the mean of the neighbouring pixels calculated with `f_neighbour_masked_sum` (once again how do we know a pixel is bad? is it static or can we infer it in some other way?)
- `f_scrub`: radiation scrubbing, calculates the mean of its 4 temporal neighbours (between -2 and +2), and if the difference is above a threashold, it replaces the pixel value with the mean
- `f_gain`: gain correction, each pixel value is multiplied by the corresponding gain right shifted by 16 places (one again, why and how?)
- `f_2x2_bin`: spatial binning, a binned frame is obtained by having each pixel value be the sum of the corresponding pixel in the frame, the two obtained by increasing one of the 2 indexes, and the one obtained by increasing both (convoluted description, check the code it is clear)
- `f_coadd`: co-adding frames, it creates the image output by adding to the value of each pixel the corresponding value in the binned frame

## Device (`ports/cpu/device.c`)


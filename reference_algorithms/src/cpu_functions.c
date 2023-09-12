#ifdef INT
typedef int bench_t;
typedef float result_bench_t;
#elif FLOAT
typedef float bench_t;
typedef float result_bench_t;
#elif DOUBLE
typedef double bench_t;
typedef double result_bench_t;
#else
typedef float bench_t;
typedef float result_bench_t;
#endif

#ifdef STD
#include <math.h>
#endif

#define M_PI 3.14159265358979323846

#define HIGHPASSFILTERSIZE 7
#define LOWPASSFILTERSIZE 9

#ifdef INT
typedef int bench_t;
#elif FLOAT
typedef float bench_t;
static const bench_t lowpass_filter_cpu[LOWPASSFILTERSIZE] = {0.037828455507, -0.023849465020, -0.110624404418, 0.377402855613, 0.852698679009, 0.377402855613, -0.110624404418, -0.023849465020, 0.037828455507};
static const bench_t highpass_filter_cpu[HIGHPASSFILTERSIZE] = {-0.064538882629, 0.040689417609, 0.418092273222, -0.788485616406, 0.418092273222, 0.040689417609, -0.064538882629};
#else
typedef double bench_t;
static const bench_t lowpass_filter_cpu[LOWPASSFILTERSIZE] = {0.037828455507, -0.023849465020, -0.110624404418, 0.377402855613, 0.852698679009, 0.377402855613, -0.110624404418, -0.023849465020, 0.037828455507};
static const bench_t highpass_filter_cpu[HIGHPASSFILTERSIZE] = {-0.064538882629, 0.040689417609, 0.418092273222, -0.788485616406, 0.418092273222, 0.040689417609, -0.064538882629};
#endif

void matrix_multiplication(const bench_t *A, const bench_t *B, bench_t *C, const unsigned int n, const unsigned int m, const unsigned int w)
{
    for (unsigned int i = 0; i < n; ++i)
    {
        for (unsigned int j = 0; j < w; ++j)
        {
            for (unsigned int k = 0; k < m; ++k)
            {
                C[i * n + j] = C[i * n + j] + A[i * n + k] * B[k * w + j];
            }
        }
    }
}

void relu(const bench_t *A, bench_t *B, const unsigned int size)
{
    for (unsigned int i = 0; i < size; ++i)
    {
        for (unsigned int j = 0; j < size; ++j)
        {
            if (A[i * size + j] > 0)
            {
                B[i * size + j] = A[i * size + j];
            }
            else
            {
                B[i * size + j];
            }
        }
    }
}

void softmax(const bench_t *A, bench_t *B, const unsigned int size)
{
    bench_t sum_values = 0;
    bench_t value = 0;
    for (unsigned int i = 0; i < size; ++i)
    {
        for (unsigned int j = 0; j < size; ++j)
        {

            value = exp(A[i * size + j]);
            sum_values += value;
            B[i * size + j] = value;
        }
    }
    for (unsigned int i = 0; i < size; ++i)
    {
        for (unsigned int j = 0; j < size; ++j)
        {
            B[i * size + j] = (B[i * size + j] / sum_values);
        }
    }
}

void matrix_convolution(const bench_t *A, bench_t *kernel, bench_t *B, const int size, const int kernel_size)
{
    // loop for the image
    int kernel_rad = kernel_size / 2;
    for (int x = 0; x < size; ++x)
    {
        for (int y = 0; y < size; ++y)
        {
            bench_t sum = 0;
            // loop over the kernel
            for (int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3
            {
                for (int j = -kernel_rad; j <= kernel_rad; ++j)
                {
                    // get value
                    bench_t value = 0;

                    if (i + x < 0 || j + y < 0)
                    {
                        value = 0;
                        // printf("ENTRO %d %d\n", i + x , j + y);
                    }
                    else if (i + x > size - 1 || j + y > size - 1)
                    {
                        value = 0;
                        // printf("ENTRO UPPER%d %d\n", i + x , j + y);
                    }
                    else
                    {
                        value = A[(x + i) * size + (y + j)];
                    }
                    // printf("ACHIVED position  %d %d value %f\n", (x + i) , (y + j), value);
                    sum += value * kernel[(i + kernel_rad) * kernel_size + (j + kernel_rad)];
                }
            }

            B[x * size + y] = sum;
        }
    }
}

// helper for max pooling
bench_t max(bench_t a, bench_t b)
{
    return (a >= b) ? a : b;
}

void max_pooling(const bench_t *A, bench_t *B, const unsigned int size, const unsigned int stride, const unsigned int lateral_stride)
{
    unsigned int stride_size = stride * stride;
    bench_t max_value = 0;
    for (unsigned int i = 0; i < size; i += stride)
    {
        for (unsigned int j = 0; j < size; j += stride)
        {
            max_value = A[i * size + j]; // init value
            // printf("init %f pos i %d, pos j %d\n", max_value, i, j);
            for (unsigned int x = 0; x < stride; ++x)
            {
                for (unsigned int y = 0; y < stride; ++y)
                {
                    // printf("max %f, value %f, pos x %d, pos y %d \n", max_value, A[(i + x) * size + (j +y)],i + x , j +y);
                    max_value = max(max_value, A[(i + x) * size + (j + y)]);
                }
            }

            B[(i / stride) * lateral_stride + (j / stride)] = max_value;
            // printf("value %f, posB x %d, posB y %d \n", B[(i / stride)* lateral_stride + (j/stride)], (i / stride) , (j/stride));
        }
    }
}

void ccsds_wavelet_transform(const bench_t *A, bench_t *B, const int size)
{
// the output will be in the B array the lower half will be the lowpass filter and the half_up will be the high pass filter
#ifdef INT
    unsigned int full_size = size * 2;
    // integer computation
    // high part
    for (unsigned int i = 0; i < size; ++i)
    {
        bench_t sum_value_high = 0;
        // specific cases
        if (i == 0)
        {
            sum_value_high = A[1] - (int)(((9.0 / 16.0) * (A[0] + A[2])) - ((1.0 / 16.0) * (A[2] + A[4])) + (1.0 / 2.0));
        }
        else if (i == size - 2)
        {
            sum_value_high = A[2 * size - 3] - (int)(((9.0 / 16.0) * (A[2 * size - 4] + A[2 * size - 2])) - ((1.0 / 16.0) * (A[2 * size - 6] + A[2 * size - 2])) + (1.0 / 2.0));
        }
        else if (i == size - 1)
        {
            sum_value_high = A[2 * size - 1] - (int)(((9.0 / 8.0) * (A[2 * size - 2])) - ((1.0 / 8.0) * (A[2 * size - 4])) + (1.0 / 2.0));
        }
        else
        {
            // generic case
            sum_value_high = A[2 * i + 1] - (int)(((9.0 / 16.0) * (A[2 * i] + A[2 * i + 2])) - ((1.0 / 16.0) * (A[2 * i - 2] + A[2 * i + 4])) + (1.0 / 2.0));
        }

        // store
        B[i + size] = sum_value_high;
    }
    // low_part
    for (unsigned int i = 0; i < size; ++i)
    {
        bench_t sum_value_low = 0;
        if (i == 0)
        {
            sum_value_low = A[0] - (int)(-(B[size] / 2.0) + (1.0 / 2.0));
        }
        else
        {
            sum_value_low = A[2 * i] - (int)(-((B[i + size - 1] + B[i + size]) / 4.0) + (1.0 / 2.0));
        }

        B[i] = sum_value_low;
    }

#else
    // flotating part
    unsigned int full_size = size * 2;
    int hi_start = -(LOWPASSFILTERSIZE / 2);
    int hi_end = LOWPASSFILTERSIZE / 2;
    int gi_start = -(HIGHPASSFILTERSIZE / 2);
    int gi_end = HIGHPASSFILTERSIZE / 2;

    for (unsigned int i = 0; i < size; ++i)
    {
        // loop over N elements of the input vector.
        bench_t sum_value_low = 0;
        // first process the lowpass filter
        for (int hi = hi_start; hi < hi_end + 1; ++hi)
        {
            int x_position = (2 * i) + hi;
            if (x_position < 0)
            {
                // turn negative to positive
                x_position = x_position * -1;
            }
            else if (x_position > full_size - 1)
            {
                x_position = full_size - 1 - (x_position - (full_size - 1));
                ;
            }
            // now I need to restore the hi value to work with the array
            sum_value_low += lowpass_filter_cpu[hi + hi_end] * A[x_position];
        }
        // store the value
        B[i] = sum_value_low;
        bench_t sum_value_high = 0;
        // second process the Highpass filter
        for (int gi = gi_start; gi < gi_end + 1; ++gi)
        {
            int x_position = (2 * i) + gi + 1;
            if (x_position < 0)
            {
                // turn negative to positive
                x_position = x_position * -1;
            }
            else if (x_position > full_size - 1)
            {
                x_position = full_size - 1 - (x_position - (full_size - 1));
            }
            sum_value_high += highpass_filter_cpu[gi + gi_end] * A[x_position];
        }
        // store the value
        B[i + size] = sum_value_high;
    }

#endif
}

void vector_convolution(const bench_t *A, bench_t *kernel, bench_t *B, const int size, const int kernel_size)
{
    const unsigned int kernel_rad = kernel_size / 2;
    const unsigned int output_size = size + kernel_size - 1;
    for (unsigned int i = 0; i < output_size; ++i)
    {
        for (unsigned int j = 0; j < kernel_size; ++j)
        {
            if (i + (j - kernel_size + 1) >= 0 && i + (j - kernel_size + 1) < size)
            {
                B[i] += kernel[kernel_size - j - 1] * A[i + (j - kernel_size + 1)];
            }
        }
    }
}

void fft_function(bench_t *data, const long nn)
{
    long n, mmax, m, j, istep, i;
    bench_t wtemp, wr, wpr, wpi, wi, theta;
    bench_t tempr, tempi;
    // reverse-binary reindexing
    n = nn << 1;
    j = 1;
    for (i = 1; i < n; i += 2)
    {
        if (j > i)
        {
            // manually substituting std::swap to use C
            bench_t tmp = data[j - 1];
            data[j - 1] = data[i - 1];
            data[i - 1] = tmp;
            tmp = data[j];
            data[j] = data[i];
            data[i] = tmp;
        }
        m = nn;
        while (m >= 2 && j > m)
        {
            j -= m;
            m >>= 1;
        }
        j += m;
    };

    // here begins the Danielson-Lanczos section
    mmax = 2;
    while (n > mmax)
    {
        istep = mmax << 1;
        theta = -(2 * M_PI / mmax);
        wtemp = sin(0.5 * theta);
        wpr = -2.0 * wtemp * wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;

        for (m = 1; m < mmax; m += 2)
        {
            for (i = m; i <= n; i += istep)
            {
                j = i + mmax;
                tempr = wr * data[j - 1] - wi * data[j];
                tempi = wr * data[j] + wi * data[j - 1];

                data[j - 1] = data[i - 1] - tempr;
                data[j] = data[i] - tempi;
                data[i - 1] += tempr;
                data[i] += tempi;
            }

            wtemp = wr;
            wr += wr * wpr - wi * wpi;
            wi += wi * wpr + wtemp * wpi;
        }
        mmax = istep;
    }
}

void aux_fft_function(bench_t* data, long nn, long start_pos){
    long loop_w = 0, loop_for_1 = 0, loop_for_2 = 0; 
    long n, mmax, m, j, istep, i , window = nn;
    bench_t wtemp, wr, wpr, wpi, wi, theta;
    bench_t tempr, tempi;
    // reverse-binary reindexing for all data 
    nn = nn>>1;

    n = nn<<1;
    //printf(" nn %ld n %ld window %ld start_pos %ld,\n",nn, n, window, start_pos);
    j=1;
    for (i=1; i<n; i+=2) {
        if (j>i) {
            // manually substituting std::swap to use C
            bench_t tmp = data[(start_pos * window) + (j-1)];
            data[(start_pos * window) + (j-1)] = data[(start_pos * window) + (i-1)];
            data[(start_pos * window) + (i-1)] = tmp;
            tmp = data[(start_pos * window) + j];
            data[(start_pos * window) + j] = data[(start_pos * window) + i];
            data[(start_pos * window) + i] = tmp;
            // std::swap(data[(start_pos * window) + (j-1)], data[(start_pos * window) + (i-1)]);
            // std::swap(data[(start_pos * window) + j], data[(start_pos * window) + i]);
        }
        m = nn;
        while (m>=2 && j>m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    };
    
    // here begins the Danielson-Lanczos section for each window
    mmax=2;
    while (n>mmax) {
        istep = mmax<<1;
        theta = -(2*M_PI/mmax);
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;

        for (m=1; m < mmax; m += 2) {
            for (i=m; i <= n; i += istep) {
                j=i+mmax;
                tempr = wr*data[(start_pos * window) + j-1] - wi*data[(start_pos * window) +j];
                tempi = wr * data[(start_pos * window) + j] + wi*data[(start_pos * window) + j-1];
                
                data[(start_pos * window) + j-1] = data[(start_pos * window) + i-1] - tempr;
                data[(start_pos * window) +j] = data[(start_pos * window) + i] - tempi;
                data[(start_pos * window) + i-1] += tempr;
                data[(start_pos * window) +i] += tempi;
                ++loop_for_1;
                //printf("wr %f wi %f\n", wr, wi);
            }
            loop_for_1 = 0;
            
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
            ++loop_for_2;

        }
        loop_for_2 = 0;
        mmax=istep;
    ++loop_w;    
    }
}


void fft_windowed_function(bench_t* data ,bench_t* output,const long window,const long nn){
    // do for all window
    for (unsigned int i = 0; i < (nn * 2 - window + 1); i+=2){
        // copy values of the  window to output
        for(unsigned int j = 0; j < window ; ++j){
            output[i * window + j] = data[i+j];
        }
        aux_fft_function(output, window, i);
    }	
}

void lrn(const bench_t* A, bench_t* B, const unsigned int size)
{
	#ifdef FLOAT
	const float K = 2;
	const float ALPHA = 10e-4;
	const float BETA = 0.75;
	#else 
	const double K = 2;
	const double ALPHA = 10e-4;
	const double BETA = 0.75;
	#endif

	for (unsigned int i = 0; i < size; ++i)
	{
		for (unsigned int j = 0; j < size; ++j)
		{
			B[i*size+j] = A[i*size+j]/pow((K+ALPHA*pow(A[i*size+j],2)),BETA);
		}
	}

}

result_bench_t get_mean_value_matrix(const bench_t* A,const int size) {
    bench_t suma_valores = 0;
    result_bench_t final_value = 0;
        
    // Be aware of precision errors.

    for (int i=0; i<size; i++){
        for (int j=0; j<size; j++){
            suma_valores += A[i*size+j];
        }
    }

    final_value = (result_bench_t) suma_valores / (result_bench_t) (size*size);
    return final_value;
}

void correlation(const bench_t *A, const bench_t *B, result_bench_t *R, const int size)
{
    result_bench_t mean_a_matrix = get_mean_value_matrix(A, size);
    result_bench_t mean_b_matrix = get_mean_value_matrix(B, size);

    // first get the final value  in A (A - mean(a)) and in B (B - mean(b))
    result_bench_t acumulate_value_a_b = 0;
    result_bench_t acumulate_value_a_a = 0;
    result_bench_t acumulate_value_b_b = 0;

    result_bench_t result_mean_a = 0;
    result_bench_t result_mean_b = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            result_mean_a = (result_bench_t) A[i * size + j] - mean_a_matrix;
            result_mean_b = (result_bench_t) B[i * size + j] - mean_b_matrix;
            acumulate_value_a_b += result_mean_a * result_mean_b;
            acumulate_value_a_a += result_mean_a * result_mean_a;
            acumulate_value_b_b += result_mean_b * result_mean_b;
        }
    }
    // final calculation
    *R = (result_bench_t)(acumulate_value_a_b / (result_bench_t)(sqrt(acumulate_value_a_a * acumulate_value_b_b)));
}
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

#include <math.h>

#define M_PI 3.14159265358979323846

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

/*
void correlation_2D(const bench_t *A, const bench_t *B, result_bench_t *R, const int size)
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
            result_mean_a = A[i * size + j] - mean_a_matrix;
            result_mean_b = B[i * size + j] - mean_b_matrix;
            acumulate_value_a_b += result_mean_a * result_mean_b;
            acumulate_value_a_a += result_mean_a * result_mean_a;
            acumulate_value_b_b += result_mean_b * result_mean_b;
        }
    }
    // final calculation
    *R = (result_bench_t)(acumulate_value_a_b / (result_bench_t)(sqrt(acumulate_value_a_a * acumulate_value_b_b)));
}

void fft_function(bench_t *data, int64_t nn)
{
    int64_t n, mmax, m, j, istep, i;
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

void aux_fft_function(bench_t *data, int64_t nn, int64_t start_pos)
{
    int64_t loop_w = 0, loop_for_1 = 0, loop_for_2 = 0;
    int64_t n, mmax, m, j, istep, i, window = nn;
    bench_t wtemp, wr, wpr, wpi, wi, theta;
    bench_t tempr, tempi;
    // reverse-binary reindexing for all data
    nn = nn >> 1;

    n = nn << 1;
    // printf(" nn %ld n %ld window %ld start_pos %ld,\n",nn, n, window, start_pos);
    j = 1;
    for (i = 1; i < n; i += 2)
    {
        if (j > i)
        {
            // manually substituting std::swap to use C
            bench_t tmp = data[(start_pos * window) + (j - 1)];
            data[(start_pos * window) + (j - 1)] = data[(start_pos * window) + (i - 1)];
            data[(start_pos * window) + (i - 1)] = tmp;

            tmp = data[(start_pos * window) + j];
            data[(start_pos * window) + j] = data[(start_pos * window) + i];
            data[(start_pos * window) + i] = tmp;
        }
        m = nn;
        while (m >= 2 && j > m)
        {
            j -= m;
            m >>= 1;
        }
        j += m;
    };

    // here begins the Danielson-Lanczos section for each window
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
                tempr = wr * data[(start_pos * window) + j - 1] - wi * data[(start_pos * window) + j];
                tempi = wr * data[(start_pos * window) + j] + wi * data[(start_pos * window) + j - 1];

                data[(start_pos * window) + j - 1] = data[(start_pos * window) + i - 1] - tempr;
                data[(start_pos * window) + j] = data[(start_pos * window) + i] - tempi;
                data[(start_pos * window) + i - 1] += tempr;
                data[(start_pos * window) + i] += tempi;
                ++loop_for_1;
                // printf("wr %f wi %f\n", wr, wi);
            }
            loop_for_1 = 0;

            wtemp = wr;
            wr += wr * wpr - wi * wpi;
            wi += wi * wpr + wtemp * wpi;
            ++loop_for_2;
        }
        loop_for_2 = 0;
        mmax = istep;
        ++loop_w;
    }
}

void fft_function_windowed(bench_t *data, bench_t *output, const int64_t window, const int64_t nn)
{
    // do for all window
    for (unsigned int i = 0; i < (nn * 2 - window + 1); i += 2)
    {
        // copy values of the  window to output
        for (unsigned int j = 0; j < window; ++j)
        {
            output[i * window + j] = data[i + j];
        }
        aux_fft_function(output, window, i);
    }
}
*/
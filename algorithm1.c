#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include "algorithm1.h"

void print128(__m128d var) {
  double val[2];
  memcpy(val, &var, sizeof(val));
  printf(" %.17g,  %.17g\n", val[0], val[1]);
}

void print256(__m256d var)
{
    double val[4];
    memcpy(val, &var, sizeof(val));
    printf("%.17g, %.17g, %.17g, %.17g\n", val[0], val[1], val[2], val[3]); 
}

double sort_total_time = 0.0f;
int sort_times = 0;

double get_distance_total_time = 0.0f;
int get_distance_times = 0;

double get_distance_basic(double* a, double* b, int length) {
    double distance = 0;
    for (int i = 0; i < length; i++) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    distance = sqrt(distance);
    return distance;
}

void sort_basic(double* a, int length, int* b)
{
    int i, j, t1;
    double t;
    for (j = 0; j < length; j++) {
        for (i = 0; i < length - 1 - j; i++) {
            if (a[i] > a[i + 1])
            {
                t = a[i];
                a[i] = a[i + 1];
                a[i + 1] = t;

                t1 = b[i];
                b[i] = b[i + 1];
                b[i + 1] = t1;
            }
        }
    }
}

void merge(double* a, int* b, int left, int medium, int right,
	double* left_temp, double* right_temp, int* index_left_temp, int* index_right_temp)
{
	int leftN = medium - left + 1;
	int rightN = right - medium;
	for (int i = 0; i < leftN; i++)
	{
		left_temp[i] = a[left + i];
		index_left_temp[i] = b[left + i];
	}

	for (int i = 0; i < rightN; i++)
	{
		right_temp[i] = a[medium + 1 + i];
		index_right_temp[i] = b[medium + 1 + i];
	}

	int l_current = 0;
	int r_current = 0;
	int merge_array_index = left;

	while (l_current < leftN && r_current < rightN)
	{
		if (left_temp[l_current] < right_temp[r_current])
		{
			a[merge_array_index] = left_temp[l_current];
			b[merge_array_index] = index_left_temp[l_current];
			merge_array_index++;
			l_current++;
		}
		else
		{
			a[merge_array_index] = right_temp[r_current];
			b[merge_array_index] = index_right_temp[r_current];
			merge_array_index++;
			r_current++;
		}
	}

	while (l_current < leftN)
	{
		a[merge_array_index] = left_temp[l_current];
		b[merge_array_index] = index_left_temp[l_current];
		merge_array_index++;
		l_current++;
	}

	while (r_current < rightN)
	{
		a[merge_array_index] = right_temp[r_current];
		b[merge_array_index] = index_right_temp[r_current];
		merge_array_index++;
		r_current++;
	}
}

void merge_sort(double* a, int* b, int left, int right, 
	double* left_temp, double* right_temp, int* index_left_temp, int* index_right_temp)
{
	int medium = (left + right) / 2;
	if (left < right)
	{
		merge_sort(a, b, left, medium, left_temp, right_temp, index_left_temp, index_right_temp);
		merge_sort(a, b, medium+1, right, left_temp, right_temp, index_left_temp, index_right_temp);
		merge(a, b, left, medium, right, left_temp, right_temp, index_left_temp, index_right_temp);
	}
}

inline void sort_optimize_merge_sort(double* a, int length, int* b)
{
	double* left_temp = (double*)malloc(sizeof(double) * length);
	double* right_temp = (double*)malloc(sizeof(double) * length);
	int* index_left_temp = (int*)malloc(sizeof(int) * length);
	int* index_right_temp = (int*)malloc(sizeof(int) * length);
    merge_sort(a, b, 0, length-1, left_temp, right_temp, index_left_temp, index_right_temp);
}

void sort_optimize(double* a, int length, int* b)
{
#ifdef MEASURE_SUBFUNCTION
    myInt64 start_sp;
    double cycles;
    start_sp = start_tsc();
#endif
            
    double* left_temp = (double*)malloc(sizeof(double) * length);
    double* right_temp = (double*)malloc(sizeof(double) * length);
    int* index_left_temp = (int*)malloc(sizeof(int) * length);
    int* index_right_temp = (int*)malloc(sizeof(int) * length);
    merge_sort(a, b, 0, length-1, left_temp, right_temp, index_left_temp, index_right_temp);

#ifdef MEASURE_SUBFUNCTION
    sort_total_time  += (double)stop_tsc(start_sp);
    sort_times += 1;
#endif
}

// Exact-SP Algorithm: Compute KNN - Baisc Version
int** get_true_KNN_basic(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length) {
     //printf("%d-%d-%d...\n", train_size, test_size, feature_length);
    int** test_knn_gt = (int**)malloc(sizeof(int*) * test_size);

    for (int i = 0; i < test_size; i++)
    {
        test_knn_gt[i] = (int*)malloc(sizeof(int) * train_size);
        for (int j = 0; j < train_size; j++) {
            test_knn_gt[i][j] = j;
        }
    }

    double* dis_arr;
    dis_arr = (double*)malloc(sizeof(double) * train_size);


    for (int i = 0; i < test_size; i++) {
        for (int i_trn = 0; i_trn < train_size; i_trn++) {
            dis_arr[i_trn] = get_distance_basic(x_trn[i_trn], x_tst[i], feature_length);
        }
        sort_basic(dis_arr, train_size, test_knn_gt[i]);
    }
    free(dis_arr);

    return test_knn_gt;
}

// Exact-SP Algorithm: Compute KNN - Optimized Version
int** get_true_KNN_optimize1(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length) {
    /* 
     *   General Description:
     *   Apply Vector Optimization on distance calculation. Still using bubble sort. 
     *
     *   Difference with previous optimized version:
     *   Apply AVX2 optimization
     */

    int** test_knn_gt = (int**)malloc(sizeof(int*) * test_size);

    for (int i = 0; i < test_size; i++)
    {
        test_knn_gt[i] = (int*)malloc(sizeof(int) * train_size);
        for (int j = 0; j < train_size; j++) {
            test_knn_gt[i][j] = j;
        }
    }

    double* dis_arr;
    dis_arr = (double*)malloc(sizeof(double) * train_size);


    for (int i = 0; i < test_size; i++) {
        for (int i_trn = 0; i_trn < train_size; i_trn++) {
            __m256d distance = _mm256_setzero_pd();

            for (int j = 0; j < feature_length; j+=4)
            {
                __m256d a_elements = _mm256_loadu_pd(x_trn[i_trn]+j);
                __m256d b_elements = _mm256_loadu_pd(x_tst[i]+j);
                __m256d a_sub_b = _mm256_sub_pd(a_elements, b_elements);
                __m256d a_sub_b_square = _mm256_mul_pd(a_sub_b, a_sub_b);
                distance = _mm256_add_pd(distance, a_sub_b_square);
            }

            __m128d distance_back = _mm256_extractf128_pd(distance, 1);
            __m128d distance_front = _mm256_extractf128_pd(distance, 0);
            __m128d half_sum = _mm_add_pd(distance_front, distance_back);
            __m128d final_sum = _mm_hadd_pd(half_sum, half_sum);

            dis_arr[i_trn] = _mm_cvtsd_f64(final_sum);

        }
        sort_basic(dis_arr, train_size, test_knn_gt[i]);
    }
    free(dis_arr);

    return test_knn_gt;
}


// Exact-SP Algorithm: Compute KNN - Optimized Version
int** get_true_KNN_optimize2(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length) {
    /* 
     *   General Description:
     *   Apply Vector Optimization on distance calculation and merge sort. 
     *
     *   Difference with previous optimized version:
     *   Change bubble sort to merge sort.
     */

    int** test_knn_gt = (int**)malloc(sizeof(int*) * test_size);

    for (int i = 0; i < test_size; i++)
    {
        test_knn_gt[i] = (int*)malloc(sizeof(int) * train_size);
        for (int j = 0; j < train_size; j++) {
            test_knn_gt[i][j] = j;
        }
    }

    double* dis_arr;
    dis_arr = (double*)malloc(sizeof(double) * train_size);


    for (int i = 0; i < test_size; i++) {
        for (int i_trn = 0; i_trn < train_size; i_trn++) {
            __m256d distance = _mm256_setzero_pd();

            for (int j = 0; j < feature_length; j+=4)
            {
                __m256d a_elements = _mm256_loadu_pd(x_trn[i_trn]+j);
                __m256d b_elements = _mm256_loadu_pd(x_tst[i]+j);
                __m256d a_sub_b = _mm256_sub_pd(a_elements, b_elements);
                __m256d a_sub_b_square = _mm256_mul_pd(a_sub_b, a_sub_b);
                distance = _mm256_add_pd(distance, a_sub_b_square);
            }

            __m128d distance_back = _mm256_extractf128_pd(distance, 1);
            __m128d distance_front = _mm256_extractf128_pd(distance, 0);
            __m128d half_sum = _mm_add_pd(distance_front, distance_back);
            __m128d final_sum = _mm_hadd_pd(half_sum, half_sum);

            dis_arr[i_trn] = _mm_cvtsd_f64(final_sum);

        }
        sort_optimize_merge_sort(dis_arr, train_size, test_knn_gt[i]);
    }
    free(dis_arr);

    return test_knn_gt;
}

// Exact-SP Algorithm: Compute KNN - Optimized Version
int** get_true_KNN_optimize3(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length) {
    /* 
     *   General Description:
     *   Apply Vector Optimization on distance calculation, merge sort, unroll training data processing * 2. 
     *
     *   Difference with previous optimized version:
     *   Unroll training data processing * 2.
     */
    int** test_knn_gt = (int**)malloc(sizeof(int*) * test_size);

    for (int i = 0; i < test_size; i++)
    {
        test_knn_gt[i] = (int*)malloc(sizeof(int) * train_size);
        for (int j = 0; j < train_size; j++) {
            test_knn_gt[i][j] = j;
        }
    }

    double* dis_arr;
    dis_arr = (double*)malloc(sizeof(double) * train_size);


    for (int i = 0; i < test_size; i++) {
        for (int i_trn = 0; i_trn < train_size; i_trn+=2) {

            __m256d distance1 = _mm256_setzero_pd();
            __m256d distance2 = _mm256_setzero_pd();
            
            double* x_tst_start = x_tst[i];
            double* x_trn_start_1 = x_trn[i_trn];
            double* x_trn_start_2 = x_trn[i_trn+1];

            for (int j = 0; j < feature_length; j+=4) {
                __m256d x_tst_i = _mm256_loadu_pd(x_tst_start);

                __m256d x_trn_1 = _mm256_loadu_pd(x_trn_start_1 + j);
                __m256d x_trn_2 = _mm256_loadu_pd(x_trn_start_2 + j);

                __m256d sub_1 = _mm256_sub_pd(x_trn_1, x_tst_i);
                __m256d sub_2 = _mm256_sub_pd(x_trn_2, x_tst_i);

                __m256d pro_1 = _mm256_mul_pd(sub_1, sub_1);
                __m256d pro_2 = _mm256_mul_pd(sub_2, sub_2);

                distance1 = _mm256_add_pd(distance1, pro_1);
                distance2 = _mm256_add_pd(distance2, pro_2);

                x_tst_start += 4;
            }

            __m128d distance_back = _mm256_extractf128_pd(distance1, 1);
            __m128d distance_front = _mm256_extractf128_pd(distance1, 0);
            __m128d half_sum = _mm_add_pd(distance_front, distance_back);
            __m128d final_sum = _mm_hadd_pd(half_sum, half_sum);

            dis_arr[i_trn] = _mm_cvtsd_f64(final_sum);

            distance_back = _mm256_extractf128_pd(distance2, 1);
            distance_front = _mm256_extractf128_pd(distance2, 0);
            half_sum = _mm_add_pd(distance_front, distance_back);
            final_sum = _mm_hadd_pd(half_sum, half_sum);

            dis_arr[i_trn+1] = _mm_cvtsd_f64(final_sum);
        }
        sort_optimize_merge_sort(dis_arr, train_size, test_knn_gt[i]);
    }
    free(dis_arr);

    return test_knn_gt;

}

// Exact-SP Algorithm: Compute KNN - Optimized Version
int** get_true_KNN_optimize4(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length) {
    /* 
     *   General Description:
     *   Apply Vector Optimization on distance calculation, merge sort, unroll training data processing * 4. 
     *
     *   Difference with previous optimized version:
     *   Change training data unrolling times from 2 to 4.
     */
    int** test_knn_gt = (int**)malloc(sizeof(int*) * test_size);

    for (int i = 0; i < test_size; i++)
    {
        test_knn_gt[i] = (int*)malloc(sizeof(int) * train_size);
        for (int j = 0; j < train_size; j++) {
            test_knn_gt[i][j] = j;
        }
    }

    double* dis_arr;
    dis_arr = (double*)malloc(sizeof(double) * train_size);


    for (int i = 0; i < test_size; i++) {
        for (int i_trn = 0; i_trn < train_size; i_trn+=4) {
            __m256d distance1 = _mm256_setzero_pd();
            __m256d distance2 = _mm256_setzero_pd();
            __m256d distance3 = _mm256_setzero_pd();
            __m256d distance4 = _mm256_setzero_pd();
            
            double* x_tst_start = x_tst[i];
            double* x_trn_start_1 = x_trn[i_trn];
            double* x_trn_start_2 = x_trn[i_trn+1];
            double* x_trn_start_3 = x_trn[i_trn+2];
            double* x_trn_start_4 = x_trn[i_trn+3];

            for (int j = 0; j < feature_length; j+=4) {
                __m256d x_tst_i = _mm256_loadu_pd(x_tst_start);

                __m256d x_trn_1 = _mm256_loadu_pd(x_trn_start_1 + j);
                __m256d x_trn_2 = _mm256_loadu_pd(x_trn_start_2 + j);
                __m256d x_trn_3 = _mm256_loadu_pd(x_trn_start_3 + j);
                __m256d x_trn_4 = _mm256_loadu_pd(x_trn_start_4 + j);

                __m256d sub_1 = _mm256_sub_pd(x_trn_1, x_tst_i);
                __m256d sub_2 = _mm256_sub_pd(x_trn_2, x_tst_i);
                __m256d sub_3 = _mm256_sub_pd(x_trn_3, x_tst_i);
                __m256d sub_4 = _mm256_sub_pd(x_trn_4, x_tst_i);

                __m256d pro_1 = _mm256_mul_pd(sub_1, sub_1);
                __m256d pro_2 = _mm256_mul_pd(sub_2, sub_2);
                __m256d pro_3 = _mm256_mul_pd(sub_3, sub_3);
                __m256d pro_4 = _mm256_mul_pd(sub_4, sub_4);

                distance1 = _mm256_add_pd(distance1, pro_1);
                distance2 = _mm256_add_pd(distance2, pro_2);
                distance3 = _mm256_add_pd(distance3, pro_3);
                distance4 = _mm256_add_pd(distance4, pro_4);

                x_tst_start += 4;
            }

            distance1 = _mm256_hadd_pd(distance1, distance1);
            distance2 = _mm256_hadd_pd(distance2, distance2);
            distance3 = _mm256_hadd_pd(distance3, distance3);
            distance4 = _mm256_hadd_pd(distance4, distance4);

            __m128d half_dist1 = _mm256_extractf128_pd(distance1, 1);
            __m128d half_dist3 = _mm256_extractf128_pd(distance3, 0);

            __m256d dist_13_1 = _mm256_insertf128_pd(distance1, half_dist3, 1);
            __m256d dist_13_2 = _mm256_insertf128_pd(distance3, half_dist1, 0);
            __m256d half_sum_13 = _mm256_add_pd(dist_13_1, dist_13_2);

            __m128d half_dist2 = _mm256_extractf128_pd(distance2, 1);
            __m128d half_dist4 = _mm256_extractf128_pd(distance4, 0);

            __m256d dist_24_1 = _mm256_insertf128_pd(distance2, half_dist4, 1);
            __m256d dist_24_2 = _mm256_insertf128_pd(distance4, half_dist2, 0);
            __m256d half_sum_24 = _mm256_add_pd(dist_24_1, dist_24_2);

            __m256d result = _mm256_unpacklo_pd(half_sum_13, half_sum_24);
            _mm256_storeu_pd(dis_arr + i_trn, result);
        }
        sort_optimize_merge_sort(dis_arr, train_size, test_knn_gt[i]);
    }
    free(dis_arr);

    return test_knn_gt;

}

// Exact-SP Algorithm: Compute KNN - Optimized Version
int** get_true_KNN_optimize5(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length) {
    /* 
     *   General Description:
     *   Apply Vector Optimization on distance calculation, merge sort, unroll training data processing * 4, unroll inner loop * 2. 
     *
     *   Difference with previous optimized version:
     *   Also unroll the inner loop (feature loop) * 2.
     */
    int** test_knn_gt = (int**)malloc(sizeof(int*) * test_size);

    for (int i = 0; i < test_size; i++)
    {
        test_knn_gt[i] = (int*)malloc(sizeof(int) * train_size);
        for (int j = 0; j < train_size; j++) {
            test_knn_gt[i][j] = j;
        }
    }

    double* dis_arr;
    dis_arr = (double*)malloc(sizeof(double) * train_size);


    for (int i = 0; i < test_size; i++) {
        for (int i_trn = 0; i_trn < train_size; i_trn+=4) {
            __m256d distance1 = _mm256_setzero_pd();
            __m256d distance2 = _mm256_setzero_pd();
            __m256d distance3 = _mm256_setzero_pd();
            __m256d distance4 = _mm256_setzero_pd();
            __m256d distance1_2 = _mm256_setzero_pd();
            __m256d distance2_2 = _mm256_setzero_pd();
            __m256d distance3_2 = _mm256_setzero_pd();
            __m256d distance4_2 = _mm256_setzero_pd();

            double* x_tst_start = x_tst[i];
            double* x_trn_start_1 = x_trn[i_trn];
            double* x_trn_start_2 = x_trn[i_trn+1];
            double* x_trn_start_3 = x_trn[i_trn+2];
            double* x_trn_start_4 = x_trn[i_trn+3];

            for (int j = 0; j < feature_length; j+=8) {
                __m256d x_tst_i = _mm256_loadu_pd(x_tst_start);
                __m256d x_tst_i_2 = _mm256_loadu_pd(x_tst_start + 4);

                __m256d x_trn_1 = _mm256_loadu_pd(x_trn_start_1 + j);
                __m256d x_trn_1_2 = _mm256_loadu_pd(x_trn_start_1 + j + 4);

                __m256d x_trn_2 = _mm256_loadu_pd(x_trn_start_2 + j);
                __m256d x_trn_2_2 = _mm256_loadu_pd(x_trn_start_2 + j + 4);

                __m256d x_trn_3 = _mm256_loadu_pd(x_trn_start_3 + j);
                __m256d x_trn_3_2 = _mm256_loadu_pd(x_trn_start_3 + j + 4);

                __m256d x_trn_4 = _mm256_loadu_pd(x_trn_start_4 + j);
                __m256d x_trn_4_2 = _mm256_loadu_pd(x_trn_start_4 + j + 4);

                __m256d sub_1 = _mm256_sub_pd(x_trn_1, x_tst_i);
                __m256d sub_1_2 = _mm256_sub_pd(x_trn_1_2, x_tst_i_2);

                __m256d sub_2 = _mm256_sub_pd(x_trn_2, x_tst_i);
                __m256d sub_2_2 = _mm256_sub_pd(x_trn_2_2, x_tst_i_2);

                __m256d sub_3 = _mm256_sub_pd(x_trn_3, x_tst_i);
                __m256d sub_3_2 = _mm256_sub_pd(x_trn_3_2, x_tst_i_2);

                __m256d sub_4 = _mm256_sub_pd(x_trn_4, x_tst_i);
                __m256d sub_4_2 = _mm256_sub_pd(x_trn_4_2, x_tst_i_2);

                __m256d pro_1 = _mm256_mul_pd(sub_1, sub_1);
                __m256d pro_1_2 = _mm256_mul_pd(sub_1_2, sub_1_2);

                __m256d pro_2 = _mm256_mul_pd(sub_2, sub_2);
                __m256d pro_2_2 = _mm256_mul_pd(sub_2_2, sub_2_2);

                __m256d pro_3 = _mm256_mul_pd(sub_3, sub_3);
                __m256d pro_3_2 = _mm256_mul_pd(sub_3_2, sub_3_2);

                __m256d pro_4 = _mm256_mul_pd(sub_4, sub_4);
                __m256d pro_4_2 = _mm256_mul_pd(sub_4_2, sub_4_2);

                distance1 = _mm256_add_pd(distance1, pro_1);
                distance1_2 = _mm256_add_pd(distance1_2, pro_1_2);

                distance2 = _mm256_add_pd(distance2, pro_2);
                distance2_2 = _mm256_add_pd(distance2_2, pro_2_2);

                distance3 = _mm256_add_pd(distance3, pro_3);
                distance3_2 = _mm256_add_pd(distance3_2, pro_3_2);

                distance4 = _mm256_add_pd(distance4, pro_4);
                distance4_2 = _mm256_add_pd(distance4_2, pro_4_2);

                x_tst_start += 8;
            }

            distance1 = _mm256_add_pd(distance1, distance1_2);
            distance1 = _mm256_hadd_pd(distance1, distance1);

            distance2 = _mm256_add_pd(distance2, distance2_2);
            distance2 = _mm256_hadd_pd(distance2, distance2);

            distance3 = _mm256_add_pd(distance3, distance3_2);
            distance3 = _mm256_hadd_pd(distance3, distance3);

            distance4 = _mm256_add_pd(distance4, distance4_2);
            distance4 = _mm256_hadd_pd(distance4, distance4);

            __m128d half_dist1 = _mm256_extractf128_pd(distance1, 1);
            __m128d half_dist3 = _mm256_extractf128_pd(distance3, 0);

            __m256d dist_13_1 = _mm256_insertf128_pd(distance1, half_dist3, 1);
            __m256d dist_13_2 = _mm256_insertf128_pd(distance3, half_dist1, 0);
            __m256d half_sum_13 = _mm256_add_pd(dist_13_1, dist_13_2);

            __m128d half_dist2 = _mm256_extractf128_pd(distance2, 1);
            __m128d half_dist4 = _mm256_extractf128_pd(distance4, 0);

            __m256d dist_24_1 = _mm256_insertf128_pd(distance2, half_dist4, 1);
            __m256d dist_24_2 = _mm256_insertf128_pd(distance4, half_dist2, 0);
            __m256d half_sum_24 = _mm256_add_pd(dist_24_1, dist_24_2);

            __m256d result = _mm256_unpacklo_pd(half_sum_13, half_sum_24);
            _mm256_storeu_pd(dis_arr + i_trn, result);
        }
        sort_optimize_merge_sort(dis_arr, train_size, test_knn_gt[i]);
    }

    free(dis_arr);

    return test_knn_gt;
}


int** get_true_KNN_optimize6(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length) {
    /* 
     *   General Description:
     *   Apply Vector Optimization on distance calculation, merge sort, unroll training data processing * 4,
     *   unroll inner loop * 2, try to improve the cache locality of test data accessing.
     *
     *   Difference with previous optimized version:
     *   Try improving cache locality of test data. (But the result is not good...)
     */
    int** test_knn_gt = (int**)malloc(sizeof(int*) * test_size);

    for (int i = 0; i < test_size; i++)
    {
        test_knn_gt[i] = (int*)malloc(sizeof(int) * train_size);
        for (int j = 0; j < train_size; j++) {
            test_knn_gt[i][j] = j;
        }
    }

    double* dis_arr;
    dis_arr = (double*)malloc(sizeof(double) * train_size);

    __m256d dis_vector_arr[18900];

    for (int i = 0; i < test_size; i++)
    {
        for (int i = 0; i < train_size; i++)
        {
            dis_vector_arr[i] = _mm256_setzero_pd();
        }
        for (int j = 0; j < feature_length; j += 8)
        {
            __m256d x_tst_1 = _mm256_loadu_pd(x_tst[i]+j);
            __m256d x_tst_2 = _mm256_loadu_pd(x_tst[i]+j+4);
            // 8 features every time, as cache block accomodates 8 doubles
            for (int i_trn = 0; i_trn < train_size; i_trn++)
            {
                __m256d x_trn_1 = _mm256_loadu_pd(x_trn[i_trn]+j);
                __m256d x_trn_2 = _mm256_loadu_pd(x_trn[i_trn]+j+4);
                __m256d sub_1 = _mm256_sub_pd(x_trn_1, x_tst_1);
                __m256d sub_2 = _mm256_sub_pd(x_trn_2, x_tst_2);
                __m256d pro_1 = _mm256_mul_pd(sub_1, sub_1);
                __m256d pro_2 = _mm256_mul_pd(sub_2, sub_2);
                __m256d distance = _mm256_add_pd(pro_1, pro_2);
                dis_vector_arr[i_trn] = _mm256_add_pd(dis_vector_arr[i_trn], distance);
            }
        }
        for (int i_trn = 0; i_trn < train_size; i_trn += 4)
        {
            __m256d distance1 = dis_vector_arr[i_trn];
            __m256d distance2 = dis_vector_arr[i_trn+1];
            __m256d distance3 = dis_vector_arr[i_trn+2];
            __m256d distance4 = dis_vector_arr[i_trn+3];
            distance1 = _mm256_hadd_pd(distance1, distance1);
            distance2 = _mm256_hadd_pd(distance2, distance2);
            distance3 = _mm256_hadd_pd(distance3, distance3);
            distance4 = _mm256_hadd_pd(distance4, distance4);

            __m128d half_dist1 = _mm256_extractf128_pd(distance1, 1);
            __m128d half_dist3 = _mm256_extractf128_pd(distance3, 0);

            __m256d dist_13_1 = _mm256_insertf128_pd(distance1, half_dist3, 1);
            __m256d dist_13_2 = _mm256_insertf128_pd(distance3, half_dist1, 0);
            __m256d half_sum_13 = _mm256_add_pd(dist_13_1, dist_13_2);

            __m128d half_dist2 = _mm256_extractf128_pd(distance2, 1);
            __m128d half_dist4 = _mm256_extractf128_pd(distance4, 0);

            __m256d dist_24_1 = _mm256_insertf128_pd(distance2, half_dist4, 1);
            __m256d dist_24_2 = _mm256_insertf128_pd(distance4, half_dist2, 0);
            __m256d half_sum_24 = _mm256_add_pd(dist_24_1, dist_24_2);

            __m256d result = _mm256_unpacklo_pd(half_sum_13, half_sum_24);
            _mm256_storeu_pd(dis_arr + i_trn, result);
        }
        sort_optimize_merge_sort(dis_arr, train_size, test_knn_gt[i]);
    }


    free(dis_arr);

    return test_knn_gt;
}

// Exact-SP Algorithm: Compute KNN - Optimized Version
int** get_true_KNN_optimize7(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length) {
    /* 
     *   General Description:
     *   Apply Vector Optimization on distance calculation, merge sort, unroll training data processing * 4,
     *   unroll inner loop * 2, FMA. 
     *
     *   Difference with previous optimized version:
     *   Change add and mul --> fma.
     *   
     *   Note: As cache locality optimization didn't provide us an improvement, we didn't include it in this implementations.
     */
    int** test_knn_gt = (int**)malloc(sizeof(int*) * test_size);

    for (int i = 0; i < test_size; i++)
    {
        test_knn_gt[i] = (int*)malloc(sizeof(int) * train_size);
        for (int j = 0; j < train_size; j++) {
            test_knn_gt[i][j] = j;
        }
    }

    double* dis_arr;
    dis_arr = (double*)malloc(sizeof(double) * train_size);


    for (int i = 0; i < test_size; i++) {
        for (int i_trn = 0; i_trn < train_size; i_trn+=4) {
            __m256d distance1 = _mm256_setzero_pd();
            __m256d distance2 = _mm256_setzero_pd();
            __m256d distance3 = _mm256_setzero_pd();
            __m256d distance4 = _mm256_setzero_pd();
            __m256d distance1_2 = _mm256_setzero_pd();
            __m256d distance2_2 = _mm256_setzero_pd();
            __m256d distance3_2 = _mm256_setzero_pd();
            __m256d distance4_2 = _mm256_setzero_pd();

            double* x_tst_start = x_tst[i];
            double* x_trn_start_1 = x_trn[i_trn];
            double* x_trn_start_2 = x_trn[i_trn+1];
            double* x_trn_start_3 = x_trn[i_trn+2];
            double* x_trn_start_4 = x_trn[i_trn+3];

            for (int j = 0; j < feature_length; j+=8) {
                __m256d x_tst_i = _mm256_loadu_pd(x_tst_start);
                __m256d x_tst_i_2 = _mm256_loadu_pd(x_tst_start + 4);

                __m256d x_trn_1 = _mm256_loadu_pd(x_trn_start_1 + j);
                __m256d x_trn_1_2 = _mm256_loadu_pd(x_trn_start_1 + j + 4);

                __m256d x_trn_2 = _mm256_loadu_pd(x_trn_start_2 + j);
                __m256d x_trn_2_2 = _mm256_loadu_pd(x_trn_start_2 + j + 4);

                __m256d x_trn_3 = _mm256_loadu_pd(x_trn_start_3 + j);
                __m256d x_trn_3_2 = _mm256_loadu_pd(x_trn_start_3 + j + 4);

                __m256d x_trn_4 = _mm256_loadu_pd(x_trn_start_4 + j);
                __m256d x_trn_4_2 = _mm256_loadu_pd(x_trn_start_4 + j + 4);

                __m256d sub_1 = _mm256_sub_pd(x_trn_1, x_tst_i);
                __m256d sub_1_2 = _mm256_sub_pd(x_trn_1_2, x_tst_i_2);

                __m256d sub_2 = _mm256_sub_pd(x_trn_2, x_tst_i);
                __m256d sub_2_2 = _mm256_sub_pd(x_trn_2_2, x_tst_i_2);

                __m256d sub_3 = _mm256_sub_pd(x_trn_3, x_tst_i);
                __m256d sub_3_2 = _mm256_sub_pd(x_trn_3_2, x_tst_i_2);

                __m256d sub_4 = _mm256_sub_pd(x_trn_4, x_tst_i);
                __m256d sub_4_2 = _mm256_sub_pd(x_trn_4_2, x_tst_i_2);

                distance1 = _mm256_fmadd_pd(sub_1, sub_1, distance1);
                distance1_2 = _mm256_fmadd_pd(sub_1_2, sub_1_2, distance1_2);

                distance2 = _mm256_fmadd_pd(sub_2, sub_2, distance2);
                distance2_2 = _mm256_fmadd_pd(sub_2_2, sub_2_2, distance2_2);

                distance3 = _mm256_fmadd_pd(sub_3, sub_3, distance3);
                distance3_2 = _mm256_fmadd_pd(sub_3_2, sub_3_2, distance3_2);

                distance4 = _mm256_fmadd_pd(sub_4, sub_4, distance4);
                distance4_2 = _mm256_fmadd_pd(sub_4_2, sub_4_2, distance4_2);

                x_tst_start += 8;
            }

            distance1 = _mm256_add_pd(distance1, distance1_2);
            distance1 = _mm256_hadd_pd(distance1, distance1);

            distance2 = _mm256_add_pd(distance2, distance2_2);
            distance2 = _mm256_hadd_pd(distance2, distance2);

            distance3 = _mm256_add_pd(distance3, distance3_2);
            distance3 = _mm256_hadd_pd(distance3, distance3);

            distance4 = _mm256_add_pd(distance4, distance4_2);
            distance4 = _mm256_hadd_pd(distance4, distance4);

            __m128d half_dist1 = _mm256_extractf128_pd(distance1, 1);
            __m128d half_dist3 = _mm256_extractf128_pd(distance3, 0);

            __m256d dist_13_1 = _mm256_insertf128_pd(distance1, half_dist3, 1);
            __m256d dist_13_2 = _mm256_insertf128_pd(distance3, half_dist1, 0);
            __m256d half_sum_13 = _mm256_add_pd(dist_13_1, dist_13_2);

            __m128d half_dist2 = _mm256_extractf128_pd(distance2, 1);
            __m128d half_dist4 = _mm256_extractf128_pd(distance4, 0);

            __m256d dist_24_1 = _mm256_insertf128_pd(distance2, half_dist4, 1);
            __m256d dist_24_2 = _mm256_insertf128_pd(distance4, half_dist2, 0);
            __m256d half_sum_24 = _mm256_add_pd(dist_24_1, dist_24_2);

            __m256d result = _mm256_unpacklo_pd(half_sum_13, half_sum_24);
            _mm256_storeu_pd(dis_arr + i_trn, result);
        }
        sort_optimize_merge_sort(dis_arr, train_size, test_knn_gt[i]);
    }

    free(dis_arr);

    return test_knn_gt;
}

int** get_true_KNN_optimize8(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length) {
    /* 
     *   General Description:
     *   Apply Vector Optimization on distance calculation, merge sort, unroll training data processing * 4,
     *   unroll inner loop * 2, FMA, rewrite distance comparation to eliminate some operations.
     *
     *   Difference with previous optimized version:
     *   Rewrite distance comparation
     *   Note: As cache locality optimization didn't provide us an improvement, we didn't include it in this implementation.
     */
    int** test_knn_gt = (int**)malloc(sizeof(int*) * test_size);

    for (int i = 0; i < test_size; i++)
    {
        test_knn_gt[i] = (int*)malloc(sizeof(int) * train_size);
        for (int j = 0; j < train_size; j++) {
            test_knn_gt[i][j] = j;
        }
    }

    double* dis_arr_trn;
    dis_arr_trn = (double*)malloc(sizeof(double) * train_size);

    double* dis_arr;
    dis_arr = (double*)malloc(sizeof(double) * train_size);

    // __m256d neg2 = _mm256_set1_pd(-2.0);
    __m256d const_coef = _mm256_set1_pd(0.5);

    for (int i_trn = 0; i_trn < train_size; i_trn+=4)
    {
        __m256d distance1 = _mm256_setzero_pd();
        __m256d distance2 = _mm256_setzero_pd();
        __m256d distance3 = _mm256_setzero_pd();
        __m256d distance4 = _mm256_setzero_pd();
        __m256d distance1_2 = _mm256_setzero_pd();
        __m256d distance2_2 = _mm256_setzero_pd();
        __m256d distance3_2 = _mm256_setzero_pd();
        __m256d distance4_2 = _mm256_setzero_pd();
        for (int j = 0; j < feature_length; j+=8)
        {
            __m256d x_trn_1 = _mm256_loadu_pd(x_trn[i_trn] + j);
            __m256d x_trn_1_2 = _mm256_loadu_pd(x_trn[i_trn] + j + 4);
            __m256d x_trn_2 = _mm256_loadu_pd(x_trn[i_trn+1] + j);
            __m256d x_trn_2_2 = _mm256_loadu_pd(x_trn[i_trn+1] + j + 4);
            __m256d x_trn_3 = _mm256_loadu_pd(x_trn[i_trn+2] + j);
            __m256d x_trn_3_2 = _mm256_loadu_pd(x_trn[i_trn+2] + j + 4);
            __m256d x_trn_4 = _mm256_loadu_pd(x_trn[i_trn+3] + j);
            __m256d x_trn_4_2 = _mm256_loadu_pd(x_trn[i_trn+3] + j + 4);


            distance1 = _mm256_fmadd_pd(x_trn_1, x_trn_1, distance1);
            distance1_2 = _mm256_fmadd_pd(x_trn_1_2, x_trn_1_2, distance1_2);
            distance2 = _mm256_fmadd_pd(x_trn_2, x_trn_2, distance2);
            distance2_2 = _mm256_fmadd_pd(x_trn_2_2, x_trn_2_2, distance2_2);
            distance3 = _mm256_fmadd_pd(x_trn_3, x_trn_3, distance3);
            distance3_2 = _mm256_fmadd_pd(x_trn_3_2, x_trn_3_2, distance3_2);
            distance4 = _mm256_fmadd_pd(x_trn_4, x_trn_4, distance4);
            distance4_2 = _mm256_fmadd_pd(x_trn_4_2, x_trn_4_2, distance4_2);
        }
        distance1 = _mm256_add_pd(distance1, distance1_2);
        distance1 = _mm256_hadd_pd(distance1, distance1);

        distance2 = _mm256_add_pd(distance2, distance2_2);
        distance2 = _mm256_hadd_pd(distance2, distance2);

        distance3 = _mm256_add_pd(distance3, distance3_2);
        distance3 = _mm256_hadd_pd(distance3, distance3);

        distance4 = _mm256_add_pd(distance4, distance4_2);
        distance4 = _mm256_hadd_pd(distance4, distance4);

        __m128d half_dist1 = _mm256_extractf128_pd(distance1, 1);
        __m128d half_dist3 = _mm256_extractf128_pd(distance3, 0);

        __m256d dist_13_1 = _mm256_insertf128_pd(distance1, half_dist3, 1);
        __m256d dist_13_2 = _mm256_insertf128_pd(distance3, half_dist1, 0);
        __m256d half_sum_13 = _mm256_add_pd(dist_13_1, dist_13_2);

        __m128d half_dist2 = _mm256_extractf128_pd(distance2, 1);
        __m128d half_dist4 = _mm256_extractf128_pd(distance4, 0);

        __m256d dist_24_1 = _mm256_insertf128_pd(distance2, half_dist4, 1);
        __m256d dist_24_2 = _mm256_insertf128_pd(distance4, half_dist2, 0);
        __m256d half_sum_24 = _mm256_add_pd(dist_24_1, dist_24_2);

        __m256d result = _mm256_unpacklo_pd(half_sum_13, half_sum_24);
        
        result = _mm256_mul_pd(const_coef, result);
        _mm256_storeu_pd(dis_arr_trn + i_trn, result);
    }

    for (int i = 0; i < test_size; i++) {
        for (int i_trn = 0; i_trn < train_size; i_trn+=4) {
            __m256d distance1 = _mm256_setzero_pd();
            __m256d distance2 = _mm256_setzero_pd();
            __m256d distance3 = _mm256_setzero_pd();
            __m256d distance4 = _mm256_setzero_pd();
            __m256d distance1_2 = _mm256_setzero_pd();
            __m256d distance2_2 = _mm256_setzero_pd();
            __m256d distance3_2 = _mm256_setzero_pd();
            __m256d distance4_2 = _mm256_setzero_pd();

            double* x_tst_start = x_tst[i];
            double* x_trn_start_1 = x_trn[i_trn];
            double* x_trn_start_2 = x_trn[i_trn+1];
            double* x_trn_start_3 = x_trn[i_trn+2];
            double* x_trn_start_4 = x_trn[i_trn+3];

            __m256d trn_const = _mm256_loadu_pd(dis_arr_trn+i_trn);

            for (int j = 0; j < feature_length; j+=8) {
                __m256d x_tst_i = _mm256_loadu_pd(x_tst_start);
                __m256d x_tst_i_2 = _mm256_loadu_pd(x_tst_start + 4);

                __m256d x_trn_1 = _mm256_loadu_pd(x_trn_start_1 + j);
                __m256d x_trn_1_2 = _mm256_loadu_pd(x_trn_start_1 + j + 4);

                __m256d x_trn_2 = _mm256_loadu_pd(x_trn_start_2 + j);
                __m256d x_trn_2_2 = _mm256_loadu_pd(x_trn_start_2 + j + 4);

                __m256d x_trn_3 = _mm256_loadu_pd(x_trn_start_3 + j);
                __m256d x_trn_3_2 = _mm256_loadu_pd(x_trn_start_3 + j + 4);

                __m256d x_trn_4 = _mm256_loadu_pd(x_trn_start_4 + j);
                __m256d x_trn_4_2 = _mm256_loadu_pd(x_trn_start_4 + j + 4);

                distance1 = _mm256_fmadd_pd(x_trn_1, x_tst_i, distance1);
                distance1_2 = _mm256_fmadd_pd(x_trn_1_2, x_tst_i_2, distance1_2);

                distance2 = _mm256_fmadd_pd(x_trn_2, x_tst_i, distance2);
                distance2_2 = _mm256_fmadd_pd(x_trn_2_2, x_tst_i_2, distance2_2);

                distance3 = _mm256_fmadd_pd(x_trn_3, x_tst_i, distance3);
                distance3_2 = _mm256_fmadd_pd(x_trn_3_2, x_tst_i_2, distance3_2);

                distance4 = _mm256_fmadd_pd(x_trn_4, x_tst_i, distance4);
                distance4_2 = _mm256_fmadd_pd(x_trn_4_2, x_tst_i_2, distance4_2);

                x_tst_start += 8;
            }

            distance1 = _mm256_add_pd(distance1, distance1_2);
            distance1 = _mm256_hadd_pd(distance1, distance1);

            distance2 = _mm256_add_pd(distance2, distance2_2);
            distance2 = _mm256_hadd_pd(distance2, distance2);

            distance3 = _mm256_add_pd(distance3, distance3_2);
            distance3 = _mm256_hadd_pd(distance3, distance3);

            distance4 = _mm256_add_pd(distance4, distance4_2);
            distance4 = _mm256_hadd_pd(distance4, distance4);

            __m128d half_dist1 = _mm256_extractf128_pd(distance1, 1);
            __m128d half_dist3 = _mm256_extractf128_pd(distance3, 0);

            __m256d dist_13_1 = _mm256_insertf128_pd(distance1, half_dist3, 1);
            __m256d dist_13_2 = _mm256_insertf128_pd(distance3, half_dist1, 0);
            __m256d half_sum_13 = _mm256_add_pd(dist_13_1, dist_13_2);

            __m128d half_dist2 = _mm256_extractf128_pd(distance2, 1);
            __m128d half_dist4 = _mm256_extractf128_pd(distance4, 0);

            __m256d dist_24_1 = _mm256_insertf128_pd(distance2, half_dist4, 1);
            __m256d dist_24_2 = _mm256_insertf128_pd(distance4, half_dist2, 0);
            __m256d half_sum_24 = _mm256_add_pd(dist_24_1, dist_24_2);

            __m256d result = _mm256_unpacklo_pd(half_sum_13, half_sum_24);
            //result = _mm256_mul_pd(neg2, result);
            //result = _mm256_add_pd(result, trn_const);

            result = _mm256_sub_pd(trn_const, result);

            _mm256_storeu_pd(dis_arr + i_trn, result);
        }
        sort_optimize_merge_sort(dis_arr, train_size, test_knn_gt[i]);
    }

    free(dis_arr);

    return test_knn_gt;
}


int** get_true_KNN_optimize(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length) {
    return get_true_KNN_optimize8(x_trn, x_tst, train_size, test_size, feature_length);
}

// Basic Version-SP
double** compute_sp_basic(double** x_trn, double* y_trn, int** test_knn_gt, double* y_tst, int K, int train_size, int test_size) {
    //printf("%d-%d...\n", train_size, test_size);
    double** sp_gt = (double**)malloc(sizeof(double*) * test_size);
    for (int i = 0; i < test_size; i++)
    {
        sp_gt[i] = (double*)malloc(sizeof(double) * train_size);
    }
    int idx = 0;
    int idx1 = 0;
    int idx2 = 0;
    int val1 = 0;
    int val2 = 0;
    int min_val = 0;
    for (int j = 0; j < test_size; j++) {
        idx = test_knn_gt[j][train_size - 1]; // an
        // compute sj-an
        if (y_trn[idx] == y_tst[j]) {
            sp_gt[j][idx] = 1.0 / train_size;
        }
        else {
            sp_gt[j][idx] = 0;
        }
        // compute sj-an-1 sj-a0
        for (int i = train_size - 2; i >= 0; i--) {
            idx1 = test_knn_gt[j][i];
            idx2 = test_knn_gt[j][i + 1];
            // min(K,i)
            if (K > i + 1) {
                min_val = i + 1;
            }
            else {
                min_val = K;
            }
            // |yai = ytest,j|
            if (y_trn[idx1] == y_tst[j]) {
                val1 = 1;
            }
            else {
                val1 = 0;
            }
            // |yai+1 = ytest,j|
            if (y_trn[idx2] == y_tst[j]) {
                val2 = 1;
            }
            else {
                val2 = 0;
            }
            sp_gt[j][idx1] = sp_gt[j][idx2] + (double)(val1 - val2) * min_val / K / (i + 1);
        }
    }
    return sp_gt;
}

// Optimised version-SP
double** compute_sp_optimize(double** x_trn, double* y_trn, int** test_knn_gt, double* y_tst, int K, int train_size, int test_size) {
    /* 
     *   Difference with previous optimized version:
     *   Remove conditional assignments
     */

    double** sp_gt = (double**)malloc(sizeof(double*) * test_size);
    for (int i = 0; i < test_size; i++)
    {
        sp_gt[i] = (double*)malloc(sizeof(double) * train_size);
    }
    int idx = 0;
    int idx1 = 0;
    int idx2 = 0;
    int val1 = 0;
    int val2 = 0;
    int min_val = 0;

    double inv_train_size = 1.0 / train_size;
    double inv_K = 1.0 / K;

    for (int j = 0; j < test_size; j++) {
        idx = test_knn_gt[j][train_size - 1]; // an
        // compute sj-an
        sp_gt[j][idx] = (y_trn[idx] == y_tst[j]) * inv_train_size;

        // compute sj-an-1 sj-a0
        for (int i = train_size - 2; i >= 0; i--) {
            idx1 = test_knn_gt[j][i];
            idx2 = test_knn_gt[j][i + 1];
            // min(K,i)
            if (K > i + 1) {
                min_val = i + 1;
            }
            else {
                min_val = K;
            }
            // |yai = ytest,j|
            val1 = (y_trn[idx1] == y_tst[j]);
            // |yai+1 = ytest,j|
            val2 = (y_trn[idx2] == y_tst[j]);
            sp_gt[j][idx1] = sp_gt[j][idx2] + (double)(val1 - val2) * min_val * inv_K / (i + 1);
        }
    }
    return sp_gt;
}

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <immintrin.h>
#include "algorithm2.h"

#define CYCLES_REQUIRED 1e10
#define DATA_SIZE 10000
#define TEST_SIZE 100
#define VAL_SIZE 1000
#define CALIBRATE true

void print_heap(HeapNode* heap) {
    int i = 0;
    for (; i < heap->Size; i++) {
        printf("%f ", heap->Arr[i]);
        printf("%d ", heap->Idx[i]);
    }
    printf("\n");
}

HeapNode* create_heap(int topk) {
    HeapNode* heap;
    heap = (HeapNode*)malloc(sizeof(HeapNode));
    heap->Arr = (double*)malloc(4 * topk * sizeof(double));
    heap->Idx = (int*)malloc(4 * topk * sizeof(int));
    heap->Size = topk;
    heap->cnt = 0;
    return heap;
}

void adjust_heap(HeapNode* heap, int root) {
    while (1) {
        int left = 2 * root + 1;
        int right = 2 * root + 2;

        if (left >= heap->Size) {
            return;
        }
        int max = left;
        if (right < heap->Size && (heap->Arr[right] > heap->Arr[left]))
            max = right;
        if (heap->Arr[root] >= heap->Arr[max])
            return;
        double temp = heap->Arr[root];
        heap->Arr[root] = heap->Arr[max];
        heap->Arr[max] = temp;
        int tempidx = heap->Idx[root];
        heap->Idx[root] = heap->Idx[max];
        heap->Idx[max] = tempidx;
        root = max;
    }
}


bool add_heap(HeapNode* heap, double node, int idx) {
    if (heap->cnt < heap->Size - 1) {
        heap->Arr[heap->cnt] = node;
        heap->Idx[heap->cnt] = idx;
        heap->cnt = heap->cnt + 1;
        return true;
    }
    else if (heap->cnt == heap->Size - 1) {
        heap->Arr[heap->cnt] = node;
        heap->Idx[heap->cnt] = idx;
        adjust_heap(heap, 0);
        heap->cnt = heap->cnt + 1;
        return true;
    }
    else if (node > heap->Arr[0]) {
        return false;
    }
    else {
        heap->Arr[0] = node;
        heap->Idx[0] = idx;
        adjust_heap(heap, 0);
        return true;
    }
}

void random_shuffle(double** a, int arr_size) {
    for (int i = 0; i < arr_size; i++) {
        int r = rand() / (RAND_MAX / (arr_size - i) + 1);
        double* tmp = a[i];
        a[i] = a[r];
        a[r] = tmp;
    }
}


void random_shuffle_idx(int* a, int arr_size) {
    for (int i = 0; i < arr_size; i++) {
        int r = rand() / (RAND_MAX / (arr_size - i) + 1);
        int tmp = a[i];
        a[i] = a[r];
        a[r] = tmp;
    }
}

void random_shuffle_op(int* a, int arr_size) {
    double rand_base = RAND_MAX / arr_size + 1;
    for (int i = 0; i < arr_size; i++) {
        int r = rand() / rand_base;
        int tmp = a[i];
        a[i] = a[r];
        a[r] = tmp;
    }
}

double get_distance_basic(double* a, double* b, int length) {
    double distance = 0;
    for (int i = 0; i < length; i++) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    distance = sqrt(distance);
    return distance;
}

double get_distance_optimize(double* a, double* b, int length) {
    double distance = 0;
    for (int i = 0; i < length; i++) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    distance = sqrt(distance);
    return distance;
}


// Basic Version-Improved MC Approach
double* imc_compute_sv_basic(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length) {
    double** diff = (double**)malloc(sizeof(double*) * test_size);
    double* sv = (double*)malloc(sizeof(double) * train_size);
    double** sp_gt = (double**)malloc(sizeof(double*) * n_perm);
    int* x_trn_idx = (int*)malloc(sizeof(double) * (train_size + 1));
    double dis;
    for (int i = 0; i < n_perm; i++)
    {
        sp_gt[i] = (double*)malloc(sizeof(double) * train_size);
    }
    bool heap_changed = false;
    for (int i = 0; i < train_size; i++)
    {
        sv[i] = 0.0;
        x_trn_idx[i] = i + 1;
    }
    for (int t = 0; t < test_size; t++)
    {
        diff[t] = (double*)malloc(sizeof(double) * (train_size + 1));
        diff[t][0] = 0.0;
    }


    for (int t = 0; t < n_perm; t++) {
        if (if_permute != 1) {
            random_shuffle_idx(x_trn_idx, train_size);
        }
        for (int m = 0; m < test_size; m++) {
            // generate random permutation            
            double last_v_sum = 0;
            double v_sum = 0;
            // initialize  max heap
            HeapNode* heap = create_heap(K);
            for (int i = 0; i < train_size; i++) {
                dis = get_distance_basic(x_trn[x_trn_idx[i] - 1], x_tst[m], feature_length);
                // insert            
                heap_changed = add_heap(heap, dis, x_trn_idx[i] - 1);
                if (heap_changed) {
                    // printf("change %d,", x_trn_idx[i] - 1);
                    int n = ((i + 1) < K) ? i + 1 : K;
                    double v = 0.0;
                    for (int k = 0; k < n; k++) {
                        int ind = heap->Idx[k];
                        // printf("ind %d,", ind);
                        if (y_trn[ind] == y_tst[m]) {
                            // printf("equal %d,", ind);
                            v += 1.0;
                        }
                    }
                    v_sum = v / n;
                    diff[m][x_trn_idx[i]] = v_sum - last_v_sum;
                    last_v_sum = v_sum;
                }
                else {
                    // printf("unchange %d %f\n", i, dis);
                    //diff[t][i + 1] = 0;
                    diff[m][x_trn_idx[i]] = 0; //diff[m][x_trn_idx[i-1]];
                }
            }
        }

        for (int sv_i = 0; sv_i < train_size; sv_i++) {
            sp_gt[t][sv_i] = 0.0;
            for (int p_i = 0; p_i < test_size; p_i++) {
                // printf("%d, %d, %f\n", p_i, sv_i, diff[p_i][sv_i]);
                sp_gt[t][sv_i] = sp_gt[t][sv_i] + diff[p_i][sv_i + 1];
            }
            sp_gt[t][sv_i] = sp_gt[t][sv_i] / test_size;
        }
    }

    double sum_sv;
    for (int i = 0; i < train_size; i++)
    {
        sum_sv = 0.0;
        for (int j = 0; j < n_perm; j++) {
            // printf("%d, %d, %f\n", j, i, sp_gt[j][i]);
            sum_sv += sp_gt[j][i];
        }
        sv[i] = sum_sv / n_perm;
    }

    return sv;
}




/*
    imc_compute_sv_optimize_v1 Get Distance Vectorization (Remove unnecessary op +Unroll + AVX2)
*/
double* imc_compute_sv_optimize_v1(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length) {
    double** diff = (double**)malloc(sizeof(double*) * test_size);
    double* sv = (double*)malloc(sizeof(double) * train_size);
    double** sp_gt = (double**)malloc(sizeof(double*) * n_perm);
    int* x_trn_idx = (int*)malloc(sizeof(double) * (train_size + 1));
    double dis;
    for (int i = 0; i < n_perm; i++)
    {
        sp_gt[i] = (double*)malloc(sizeof(double) * train_size);
    }
    bool heap_changed = false;
    for (int i = 0; i < train_size; i++)
    {
        sv[i] = 0.0;
        x_trn_idx[i] = i + 1;
    }
    for (int t = 0; t < test_size; t++)
    {
        diff[t] = (double*)malloc(sizeof(double) * (train_size + 1));
        diff[t][0] = 0.0;
    }


    double* dis_arr;
    dis_arr = (double*)malloc(sizeof(double) * train_size);


    for (int t = 0; t < n_perm; t++) {
        if (if_permute != 1) {
            random_shuffle_idx(x_trn_idx, train_size);
        }
        for (int m = 0; m < test_size; m++) {
            // generate random permutation            
            double last_v_sum = 0;
            double v_sum = 0;
            // initialize  max heap
            HeapNode* heap = create_heap(K);
            for (int i = 0; i < train_size; i += 4) {

                //[Opimize Before] dis = get_distance_basic(x_trn[x_trn_idx[i] - 1], x_tst[m], feature_length);
                __m256d distance1 = _mm256_setzero_pd();
                __m256d distance2 = _mm256_setzero_pd();
                __m256d distance3 = _mm256_setzero_pd();
                __m256d distance4 = _mm256_setzero_pd();
                __m256d distance1_2 = _mm256_setzero_pd();
                __m256d distance2_2 = _mm256_setzero_pd();
                __m256d distance3_2 = _mm256_setzero_pd();
                __m256d distance4_2 = _mm256_setzero_pd();

                double* x_tst_start = x_tst[m];
                double* x_trn_start_1 = x_trn[x_trn_idx[i] - 1];
                double* x_trn_start_2 = x_trn[x_trn_idx[i + 1] - 1];
                double* x_trn_start_3 = x_trn[x_trn_idx[i + 2] - 1];
                double* x_trn_start_4 = x_trn[x_trn_idx[i + 3] - 1];

                for (int j = 0; j < feature_length; j += 8) {
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
                _mm256_storeu_pd(dis_arr + i, result);


                //dis_arr[i] = get_distance_basic(x_trn[x_trn_idx[i] - 1], x_tst[m], feature_length);
                //dis_arr[i+1] = get_distance_basic(x_trn[x_trn_idx[i+1] - 1], x_tst[m], feature_length);
                //dis_arr[i+2] = get_distance_basic(x_trn[x_trn_idx[i+2] - 1], x_tst[m], feature_length);
                //dis_arr[i+3] = get_distance_basic(x_trn[x_trn_idx[i+3] - 1], x_tst[m], feature_length);
            }


            for (int i = 0; i < train_size; i += 1) {
                dis = dis_arr[i];
                // printf("dis %f\n", dis);
                // insert            
                heap_changed = add_heap(heap, dis, x_trn_idx[i] - 1);
                if (heap_changed) {
                    // printf("change %d,", x_trn_idx[i] - 1);
                    int n = ((i + 1) < K) ? i + 1 : K;
                    double v = 0.0;
                    for (int k = 0; k < n; k++) {
                        int ind = heap->Idx[k];
                        // printf("ind %d,", ind);
                        if (y_trn[ind] == y_tst[m]) {
                            // printf("equal %d,", ind);
                            v += 1.0;
                        }
                    }
                    v_sum = v / n;
                    diff[m][x_trn_idx[i]] = v_sum - last_v_sum;
                    last_v_sum = v_sum;
                }
                else {
                    // printf("unchange %d %f\n", i, dis);
                    //diff[t][i + 1] = 0;
                    diff[m][x_trn_idx[i]] = 0; //diff[m][x_trn_idx[i-1]];
                }
            }
        }

        for (int sv_i = 0; sv_i < train_size; sv_i++) {
            sp_gt[t][sv_i] = 0.0;
            for (int p_i = 0; p_i < test_size; p_i++) {
                // printf("%d, %d, %f\n", p_i, sv_i, diff[p_i][sv_i]);
                sp_gt[t][sv_i] = sp_gt[t][sv_i] + diff[p_i][sv_i + 1];
            }
            sp_gt[t][sv_i] = sp_gt[t][sv_i] / test_size;
        }
    }

    double sum_sv;
    for (int i = 0; i < train_size; i++)
    {
        sum_sv = 0.0;
        for (int j = 0; j < n_perm; j++) {
            // printf("%d, %d, %f\n", j, i, sp_gt[j][i]);
            sum_sv += sp_gt[j][i];
        }
        sv[i] = sum_sv / n_perm;
    }

    return sv;
}

/*
    imc_compute_sv_optimize_v2 Compute SP Vectorization (Unroll + AVX2)
*/

double* imc_compute_sv_optimize_v2(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length) {
    double** diff = (double**)malloc(sizeof(double*) * test_size);
    double* sv = (double*)malloc(sizeof(double) * train_size);
    double** sp_gt = (double**)malloc(sizeof(double*) * n_perm);
    int* x_trn_idx = (int*)malloc(sizeof(double) * (train_size + 1));
    double dis;
    for (int i = 0; i < n_perm; i++)
    {
        sp_gt[i] = (double*)malloc(sizeof(double) * train_size);
    }
    bool heap_changed = false;
    for (int i = 0; i < train_size; i++)
    {
        sv[i] = 0.0;
        x_trn_idx[i] = i + 1;
    }
    for (int t = 0; t < test_size; t++)
    {
        diff[t] = (double*)malloc(sizeof(double) * (train_size + 1));
        diff[t][0] = 0.0;
    }


    double* dis_arr;
    dis_arr = (double*)malloc(sizeof(double) * train_size);
    double div_test_size = 1.0 / (double)test_size;
    __m256d div_vector = _mm256_set_pd(div_test_size, div_test_size, div_test_size, div_test_size);


    for (int t = 0; t < n_perm; t++) {
        if (if_permute != 1) {
            random_shuffle_idx(x_trn_idx, train_size);
        }
        for (int m = 0; m < test_size; m++) {
            // generate random permutation            
            double last_v_sum = 0;
            double v_sum = 0;
            // initialize  max heap
            HeapNode* heap = create_heap(K);
            for (int i = 0; i < train_size; i += 4) {

                //[Opimize Before] dis = get_distance_basic(x_trn[x_trn_idx[i] - 1], x_tst[m], feature_length);
                __m256d distance1 = _mm256_setzero_pd();
                __m256d distance2 = _mm256_setzero_pd();
                __m256d distance3 = _mm256_setzero_pd();
                __m256d distance4 = _mm256_setzero_pd();
                __m256d distance1_2 = _mm256_setzero_pd();
                __m256d distance2_2 = _mm256_setzero_pd();
                __m256d distance3_2 = _mm256_setzero_pd();
                __m256d distance4_2 = _mm256_setzero_pd();

                double* x_tst_start = x_tst[m];
                double* x_trn_start_1 = x_trn[x_trn_idx[i] - 1];
                double* x_trn_start_2 = x_trn[x_trn_idx[i + 1] - 1];
                double* x_trn_start_3 = x_trn[x_trn_idx[i + 2] - 1];
                double* x_trn_start_4 = x_trn[x_trn_idx[i + 3] - 1];

                for (int j = 0; j < feature_length; j += 8) {
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
                _mm256_storeu_pd(dis_arr + i, result);


                //dis_arr[i] = get_distance_basic(x_trn[x_trn_idx[i] - 1], x_tst[m], feature_length);
                //dis_arr[i+1] = get_distance_basic(x_trn[x_trn_idx[i+1] - 1], x_tst[m], feature_length);
                //dis_arr[i+2] = get_distance_basic(x_trn[x_trn_idx[i+2] - 1], x_tst[m], feature_length);
                //dis_arr[i+3] = get_distance_basic(x_trn[x_trn_idx[i+3] - 1], x_tst[m], feature_length);
            }


            for (int i = 0; i < train_size; i += 1) {
                dis = dis_arr[i];
                // printf("dis %f\n", dis);
                // insert            
                heap_changed = add_heap(heap, dis, x_trn_idx[i] - 1);
                if (heap_changed) {
                    // printf("change %d,", x_trn_idx[i] - 1);
                    int n = ((i + 1) < K) ? i + 1 : K;
                    double v = 0.0;
                    for (int k = 0; k < n; k++) {
                        int ind = heap->Idx[k];
                        // printf("ind %d,", ind);
                        if (y_trn[ind] == y_tst[m]) {
                            // printf("equal %d,", ind);
                            v += 1.0;
                        }
                    }
                    v_sum = v / n;
                    diff[m][x_trn_idx[i]] = v_sum - last_v_sum;
                    last_v_sum = v_sum;
                }
                else {
                    // printf("unchange %d %f\n", i, dis);
                    //diff[t][i + 1] = 0;
                    diff[m][x_trn_idx[i]] = 0; //diff[m][x_trn_idx[i-1]];
                }
            }
        }

        for (int sv_i = 0; sv_i < train_size; sv_i += 4) {
            double* sp_start = sp_gt[t] + sv_i;
            __m256d sp0 = _mm256_setzero_pd();
            __m256d sp1 = _mm256_setzero_pd();
            __m256d sp2 = _mm256_setzero_pd();
            __m256d sp3 = _mm256_setzero_pd();
            __m256d sp = _mm256_setzero_pd();
            for (int p_i = 0; p_i < test_size; p_i += 4) {
                // printf("%d, %d, %f\n", p_i, sv_i, diff[p_i][sv_i]);
                // __m256d diff_1 = _mm256_loadu_pd(diff_start);
                // __m256d sp_start = _mm256_loadu_pd(x_tst_start + 4);
                // double* diff_start = diff[p_i] + sv_i;
                __m256d diff_i_0 = _mm256_loadu_pd(diff[p_i] + sv_i + 1);
                __m256d diff_i_1 = _mm256_loadu_pd(diff[p_i + 1] + sv_i + 1);
                __m256d diff_i_2 = _mm256_loadu_pd(diff[p_i + 2] + sv_i + 1);
                __m256d diff_i_3 = _mm256_loadu_pd(diff[p_i + 3] + sv_i + 1);
                sp0 = _mm256_add_pd(sp0, diff_i_0);
                sp1 = _mm256_add_pd(sp1, diff_i_1);
                sp2 = _mm256_add_pd(sp2, diff_i_2);
                sp3 = _mm256_add_pd(sp3, diff_i_3);
                /*
                sum00 = sum00 + diff[p_i][sv_i + 1];
                sum01 = sum01 + diff[p_i + 1][sv_i + 1];
                sum02 = sum02 + diff[p_i + 2][sv_i + 1];
                sum03 = sum03 + diff[p_i + 3][sv_i + 1];
                sum10 = sum10 + diff[p_i][sv_i + 2];
                sum11 = sum11 + diff[p_i + 1][sv_i + 2];
                sum12 = sum12 + diff[p_i + 2][sv_i + 2];
                sum13 = sum13 + diff[p_i + 3][sv_i + 2];
                sum20 = sum20 + diff[p_i][sv_i + 3];
                sum21 = sum21 + diff[p_i + 1][sv_i + 3];
                sum22 = sum22 + diff[p_i + 2][sv_i + 3];
                sum23 = sum23 + diff[p_i + 3][sv_i + 3];
                sum30 = sum30 + diff[p_i][sv_i + 4];
                sum31 = sum31 + diff[p_i + 1][sv_i + 4];
                sum32 = sum32 + diff[p_i + 2][sv_i + 4];
                sum33 = sum33 + diff[p_i + 3][sv_i + 4];
                */
                // //[Opimize Before] sp_gt[t][sv_i] = sp_gt[t][sv_i] + diff[p_i][sv_i + 1];
            }
            sp0 = _mm256_add_pd(sp0, sp2);
            sp1 = _mm256_add_pd(sp1, sp3);
            sp = _mm256_add_pd(sp0, sp1);
            sp = _mm256_mul_pd(sp, div_vector);
            /*
            sum0 = sum00 + sum01 + sum02 + sum03;
            sum1 = sum10 + sum11 + sum12 + sum13;
            sum2 = sum20 + sum21 + sum22 + sum23;
            sum3 = sum30 + sum31 + sum32 + sum33;
            sum0 = sum0 * div_test_size;
            sum1 = sum1 * div_test_size;
            sum2 = sum2 * div_test_size;
            sum3 = sum3 * div_test_size;
            sp_gt[t][sv_i] = sum0;
            sp_gt[t][sv_i + 1] = sum1;
            sp_gt[t][sv_i + 2] = sum2;
            sp_gt[t][sv_i + 3] = sum3;
            */
            _mm256_storeu_pd(sp_start, sp);
        }


        /* unroll version
        for (int sv_i = 0; sv_i < train_size; sv_i += 4) {
            double sum0 = 0.0;
            double sum1 = 0.0;
            double sum2 = 0.0;
            double sum3 = 0.0;
            double sum00 = 0.0;
            double sum01 = 0.0;
            double sum02 = 0.0;
            double sum03 = 0.0;
            double sum10 = 0.0;
            double sum11 = 0.0;
            double sum12 = 0.0;
            double sum13 = 0.0;
            double sum20 = 0.0;
            double sum21 = 0.0;
            double sum22 = 0.0;
            double sum23 = 0.0;
            double sum30 = 0.0;
            double sum31 = 0.0;
            double sum32 = 0.0;
            double sum33 = 0.0;
            //double* sp_start = sp_gt[t][0];
            for (int p_i = 0; p_i < test_size; p_i+= 4) {

                // printf("%d, %d, %f\n", p_i, sv_i, diff[p_i][sv_i]);
                // __m256d diff_1 = _mm256_loadu_pd(diff_start);
                // __m256d sp_start = _mm256_loadu_pd(x_tst_start + 4);
                sum00 = sum00 + diff[p_i][sv_i + 1];
                sum01 = sum01 + diff[p_i + 1][sv_i + 1];
                sum02 = sum02 + diff[p_i + 2][sv_i + 1];
                sum03 = sum03 + diff[p_i + 3][sv_i + 1];
                sum10 = sum10 + diff[p_i][sv_i + 2];
                sum11 = sum11 + diff[p_i + 1][sv_i + 2];
                sum12 = sum12 + diff[p_i + 2][sv_i + 2];
                sum13 = sum13 + diff[p_i + 3][sv_i + 2];
                sum20 = sum20 + diff[p_i][sv_i + 3];
                sum20 = sum20 + diff[p_i][sv_i + 3];
                sum21 = sum21 + diff[p_i + 1][sv_i + 3];
                sum22 = sum22 + diff[p_i + 2][sv_i + 3];
                sum23 = sum23 + diff[p_i + 3][sv_i + 3];
                sum30 = sum30 + diff[p_i][sv_i + 4];
                sum31 = sum31 + diff[p_i + 1][sv_i + 4];
                sum32 = sum32 + diff[p_i + 2][sv_i + 4];
                sum33 = sum33 + diff[p_i + 3][sv_i + 4];
                // //[Opimize Before] sp_gt[t][sv_i] = sp_gt[t][sv_i] + diff[p_i][sv_i + 1];
            }
            sum0 = sum00 + sum01 + sum02 + sum03;
            sum1 = sum10 + sum11 + sum12 + sum13;
            sum2 = sum20 + sum21 + sum22 + sum23;
            sum3 = sum30 + sum31 + sum32 + sum33;
            sum0 = sum0 * div_test_size;
            sum1 = sum1 * div_test_size;
            sum2 = sum2 * div_test_size;
            sum3 = sum3 * div_test_size;
            sp_gt[t][sv_i] = sum0;
            sp_gt[t][sv_i + 1] = sum1;
            sp_gt[t][sv_i + 2] = sum2;
            sp_gt[t][sv_i + 3] = sum3;
        }
        */
    }

    double sum_sv;
    for (int i = 0; i < train_size; i++)
    {
        sum_sv = 0.0;
        for (int j = 0; j < n_perm; j++) {
            // printf("%d, %d, %f\n", j, i, sp_gt[j][i]);
            sum_sv += sp_gt[j][i];
        }
        sv[i] = sum_sv / n_perm;
    }

    return sv;
}

/*
    v3: Precomputation of Distance
*/

// Optimized Version-Improved MC Approach
double* imc_compute_sv_optimize_v3(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length) {
    double** diff = (double**)malloc(sizeof(double*) * test_size);
    double* sv = (double*)malloc(sizeof(double) * train_size);
    double** sp_gt = (double**)malloc(sizeof(double*) * n_perm);
    int* x_trn_idx = (int*)malloc(sizeof(double) * (train_size + 1));
    double dis;
    for (int i = 0; i < n_perm; i++)
    {
        sp_gt[i] = (double*)malloc(sizeof(double) * train_size);
    }
    bool heap_changed = false;
    for (int i = 0; i < train_size; i++)
    {
        sv[i] = 0.0;
        x_trn_idx[i] = i + 1;
    }
    for (int t = 0; t < test_size; t++)
    {
        diff[t] = (double*)malloc(sizeof(double) * (train_size + 1));
        diff[t][0] = 0.0;
    }


    double** dis_arr = (double**)malloc(sizeof(double*) * test_size);
    for (int t = 0; t < test_size; t++)
    {
        dis_arr[t] = (double*)malloc(sizeof(double) * train_size);
    }

    // dis_arr = (double*)malloc(sizeof(double) * train_size);
    double div_test_size = 1.0 / (double)test_size;
    __m256d div_vector = _mm256_set_pd(div_test_size, div_test_size, div_test_size, div_test_size);

    for (int m = 0; m < test_size; m++) {
        double* dis_start = dis_arr[m];
        for (int i = 0; i < train_size; i += 4) {
            //[Opimize Before] dis = get_distance_basic(x_trn[x_trn_idx[i] - 1], x_tst[m], feature_length);
            __m256d distance1 = _mm256_setzero_pd();
            __m256d distance2 = _mm256_setzero_pd();
            __m256d distance3 = _mm256_setzero_pd();
            __m256d distance4 = _mm256_setzero_pd();
            __m256d distance1_2 = _mm256_setzero_pd();
            __m256d distance2_2 = _mm256_setzero_pd();
            __m256d distance3_2 = _mm256_setzero_pd();
            __m256d distance4_2 = _mm256_setzero_pd();

            double* x_tst_start = x_tst[m];
            double* x_trn_start_1 = x_trn[i];
            double* x_trn_start_2 = x_trn[i + 1];
            double* x_trn_start_3 = x_trn[i + 2];
            double* x_trn_start_4 = x_trn[i + 3];

            for (int j = 0; j < feature_length; j += 8) {
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
            _mm256_storeu_pd(dis_start + i, result);
        }
    }

    for (int t = 0; t < n_perm; t++) {
        if (if_permute != 1) {
            random_shuffle_idx(x_trn_idx, train_size);
        }
        for (int m = 0; m < test_size; m++) {
            // generate random permutation
            double last_v_sum = 0;
            double v_sum = 0;
            // initialize  max heap
            HeapNode* heap = create_heap(K);

            for (int i = 0; i < train_size; i += 1) {
                dis = dis_arr[m][i];
                // printf("1 dis %f\n", dis);
                // printf("dis %f\n", dis);
                // insert
                heap_changed = add_heap(heap, dis, x_trn_idx[i] - 1);
                if (heap_changed) {
                    // printf("change %d,", x_trn_idx[i] - 1);
                    int n = ((i + 1) < K) ? i + 1 : K;
                    double v = 0.0;
                    for (int k = 0; k < n; k++) {
                        int ind = heap->Idx[k];
                        // printf("ind %d,", ind);
                        if (y_trn[ind] == y_tst[m]) {
                            // printf("equal %d,", ind);
                            v += 1.0;
                        }
                    }
                    v_sum = v / n;
                    diff[m][x_trn_idx[i]] = v_sum - last_v_sum;
                    last_v_sum = v_sum;
                }
                else {
                    // printf("unchange %d %f\n", i, dis);
                    //diff[t][i + 1] = 0;
                    diff[m][x_trn_idx[i]] = 0; //diff[m][x_trn_idx[i-1]];
                }
            }
        }


        for (int sv_i = 0; sv_i < train_size; sv_i += 4) {
            double* sp_start = sp_gt[t] + sv_i;
            __m256d sp0 = _mm256_setzero_pd();
            __m256d sp1 = _mm256_setzero_pd();
            __m256d sp2 = _mm256_setzero_pd();
            __m256d sp3 = _mm256_setzero_pd();
            __m256d sp = _mm256_setzero_pd();
            for (int p_i = 0; p_i < test_size; p_i += 4) {
                // printf("%d, %d, %f\n", p_i, sv_i, diff[p_i][sv_i]);
                // __m256d diff_1 = _mm256_loadu_pd(diff_start);
                // __m256d sp_start = _mm256_loadu_pd(x_tst_start + 4);
                // double* diff_start = diff[p_i] + sv_i;
                __m256d diff_i_0 = _mm256_loadu_pd(diff[p_i] + sv_i + 1);
                __m256d diff_i_1 = _mm256_loadu_pd(diff[p_i + 1] + sv_i + 1);
                __m256d diff_i_2 = _mm256_loadu_pd(diff[p_i + 2] + sv_i + 1);
                __m256d diff_i_3 = _mm256_loadu_pd(diff[p_i + 3] + sv_i + 1);
                sp0 = _mm256_add_pd(sp0, diff_i_0);
                sp1 = _mm256_add_pd(sp1, diff_i_1);
                sp2 = _mm256_add_pd(sp2, diff_i_2);
                sp3 = _mm256_add_pd(sp3, diff_i_3);
                // //[Opimize Before] sp_gt[t][sv_i] = sp_gt[t][sv_i] + diff[p_i][sv_i + 1];
            }
            sp0 = _mm256_add_pd(sp0, sp2);
            sp1 = _mm256_add_pd(sp1, sp3);
            sp = _mm256_add_pd(sp0, sp1);
            sp = _mm256_mul_pd(sp, div_vector);
            _mm256_storeu_pd(sp_start, sp);
        }

    }

    double sum_sv;
    for (int i = 0; i < train_size; i++)
    {
        sum_sv = 0.0;
        for (int j = 0; j < n_perm; j++) {
            // printf("%d, %d, %f\n", j, i, sp_gt[j][i]);
            sum_sv += sp_gt[j][i];
        }
        sv[i] = sum_sv / n_perm;
    }

    return sv;
}

/*
    v4: Heap Inline + Replace Heap Struct with Array + Reduce Div to Mult
*/

// Optimized Version-Improved MC Approach
double* imc_compute_sv_optimize_v4(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length) {
    double** diff = (double**)malloc(sizeof(double*) * test_size);
    double* sv = (double*)malloc(sizeof(double) * train_size);
    double** sp_gt = (double**)malloc(sizeof(double*) * n_perm);
    int* x_trn_idx = (int*)malloc(sizeof(double) * (train_size + 1));
    double dis;
    for (int i = 0; i < n_perm; i++)
    {
        sp_gt[i] = (double*)malloc(sizeof(double) * train_size);
    }
    bool heap_changed = false;
    for (int i = 0; i < train_size; i++)
    {
        sv[i] = 0.0;
        x_trn_idx[i] = i + 1;
    }
    for (int t = 0; t < test_size; t++)
    {
        diff[t] = (double*)malloc(sizeof(double) * (train_size + 1));
        diff[t][0] = 0.0;
    }


    double** dis_arr = (double**)malloc(sizeof(double*) * test_size);
    for (int t = 0; t < test_size; t++)
    {
        dis_arr[t] = (double*)malloc(sizeof(double) * train_size);
    }

    double div_test_size = 1.0 / (double)test_size;
    double K_divider = 1.0 / (double)K;
    double min_divider;
    __m256d div_vector = _mm256_set_pd(div_test_size, div_test_size, div_test_size, div_test_size);

    for (int m = 0; m < test_size; m++) {
        double* dis_start = dis_arr[m];
        for (int i = 0; i < train_size; i += 4) {
            //[Opimize Before] dis = get_distance_basic(x_trn[x_trn_idx[i] - 1], x_tst[m], feature_length);
            __m256d distance1 = _mm256_setzero_pd();
            __m256d distance2 = _mm256_setzero_pd();
            __m256d distance3 = _mm256_setzero_pd();
            __m256d distance4 = _mm256_setzero_pd();
            __m256d distance1_2 = _mm256_setzero_pd();
            __m256d distance2_2 = _mm256_setzero_pd();
            __m256d distance3_2 = _mm256_setzero_pd();
            __m256d distance4_2 = _mm256_setzero_pd();

            double* x_tst_start = x_tst[m];
            double* x_trn_start_1 = x_trn[i];
            double* x_trn_start_2 = x_trn[i + 1];
            double* x_trn_start_3 = x_trn[i + 2];
            double* x_trn_start_4 = x_trn[i + 3];

            for (int j = 0; j < feature_length; j += 8) {
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
            _mm256_storeu_pd(dis_start + i, result);
        }
    }


    for (int t = 0; t < n_perm; t++) {
        if (if_permute != 1) {
            random_shuffle_op(x_trn_idx, train_size);
        }

        for (int m = 0; m < test_size; m++) {
            // generate random permutation
            double last_v_sum = 0;
            double v_sum = 0;
            // initialize  max heap
            // HeapNode* heap = create_heap(K); 
            // [Optimize ]Heap Inlining
            double* heapArr = (double*)malloc(4 * K * sizeof(double));
            int* heapIdx = (int*)malloc(4 * K * sizeof(int));
            int heapSize = K;
            int heapcnt = 0;

            for (int i = 0; i < train_size; i += 1) {
                dis = dis_arr[m][i];
                // insert
                // [Optimize ]Heap Inlining
                // heap_changed = add_heap(heap, dis, x_trn_idx[i] - 1);
                double node = dis;
                int idx = x_trn_idx[i] - 1;
                if (heapcnt < heapSize - 1) {
                    heapArr[heapcnt] = node;
                    heapIdx[heapcnt] = idx;
                    heapcnt = heapcnt + 1;
                    // if heap changed true
                    // int n = ((i + 1) < K) ? i + 1 : K;
                    int n;
                    if (i > K) {
                        min_divider = K_divider;
                        n = K;
                    }
                    else {
                        min_divider = 1.0 / (double)(i + 1);
                        n = i + 1;
                    }

                    double v = 0.0;
                    for (int k = 0; k < n; k++) {
                        int ind = heapIdx[k];
                        if (y_trn[ind] == y_tst[m]) {
                            v += 1.0;
                        }
                    }
                    v_sum = v * min_divider;
                    diff[m][idx + 1] = v_sum - last_v_sum;
                    last_v_sum = v_sum;
                }
                else if (heapcnt == heapSize - 1) {
                    heapArr[heapcnt] = node;
                    heapIdx[heapcnt] = idx;
                    int root = 0;
                    int adjust_flag = 1;
                    while (adjust_flag == 1) {
                        int left = 2 * root + 1;
                        int right = 2 * root + 2;
                        if (left >= heapSize) {
                            adjust_flag = 0;
                            break;
                        }
                        int max = left;
                        if (right < heapSize && (heapArr[right] > heapArr[left]))
                            max = right;
                        if (heapArr[root] >= heapArr[max]) {
                            adjust_flag = 0;
                            break;
                        }
                        double temp = heapArr[root];
                        heapArr[root] = heapArr[max];
                        heapArr[max] = temp;
                        int tempidx = heapIdx[root];
                        heapIdx[root] = heapIdx[max];
                        heapIdx[max] = tempidx;
                        root = max;
                    }
                    // if heap changed true
                    heapcnt = heapcnt + 1;
                    // int n = ((i + 1) < K) ? i + 1 : K;
                    int n;
                    if (i > K) {
                        min_divider = K_divider;
                        n = K;
                    }
                    else {
                        min_divider = 1.0 / (double)(i + 1);
                        n = i + 1;
                    }
                    double v = 0.0;
                    for (int k = 0; k < n; k++) {
                        int ind = heapIdx[k];
                        if (y_trn[ind] == y_tst[m]) {
                            v += 1.0;
                        }
                    }
                    v_sum = v * min_divider;
                    diff[m][idx + 1] = v_sum - last_v_sum;
                    last_v_sum = v_sum;
                }
                else if (node > heapArr[0]) {
                    diff[m][idx + 1] = 0;
                }
                else {
                    heapArr[0] = node;
                    heapIdx[0] = idx;
                    int root = 0;
                    int adjust_flag = 1;
                    while (adjust_flag == 1) {
                        int left = 2 * root + 1;
                        int right = 2 * root + 2;
                        if (left >= heapSize) {
                            adjust_flag = 0;
                            break;
                        }
                        int max = left;
                        if (right < heapSize && (heapArr[right] > heapArr[left]))
                            max = right;
                        if (heapArr[root] >= heapArr[max]) {
                            adjust_flag = 0;
                            break;
                        }
                        double temp = heapArr[root];
                        heapArr[root] = heapArr[max];
                        heapArr[max] = temp;
                        int tempidx = heapIdx[root];
                        heapIdx[root] = heapIdx[max];
                        heapIdx[max] = tempidx;
                        root = max;
                    }
                    // if heap changed true
                    // int n = ((i + 1) < K) ? i + 1 : K;
                    int n;
                    if (i > K) {
                        min_divider = K_divider;
                        n = K;
                    }
                    else {
                        min_divider = 1.0 / (double)(i + 1);
                        n = i + 1;
                    }
                    double v = 0.0;
                    for (int k = 0; k < n; k++) {
                        int ind = heapIdx[k];
                        if (y_trn[ind] == y_tst[m]) {
                            v += 1.0;
                        }
                    }
                    v_sum = v * min_divider;
                    diff[m][idx + 1] = v_sum - last_v_sum;
                    last_v_sum = v_sum;
                }
            }

            free(heapArr);
            free(heapIdx);

        }

        for (int sv_i = 0; sv_i < train_size; sv_i += 4) {
            double* sp_start = sp_gt[t] + sv_i;
            __m256d sp0 = _mm256_setzero_pd();
            __m256d sp1 = _mm256_setzero_pd();
            __m256d sp2 = _mm256_setzero_pd();
            __m256d sp3 = _mm256_setzero_pd();
            __m256d sp = _mm256_setzero_pd();
            for (int p_i = 0; p_i < test_size; p_i += 4) {
                // printf("%d, %d, %f\n", p_i, sv_i, diff[p_i][sv_i]);
                __m256d diff_i_0 = _mm256_loadu_pd(diff[p_i] + sv_i + 1);
                __m256d diff_i_1 = _mm256_loadu_pd(diff[p_i + 1] + sv_i + 1);
                __m256d diff_i_2 = _mm256_loadu_pd(diff[p_i + 2] + sv_i + 1);
                __m256d diff_i_3 = _mm256_loadu_pd(diff[p_i + 3] + sv_i + 1);
                sp0 = _mm256_add_pd(sp0, diff_i_0);
                sp1 = _mm256_add_pd(sp1, diff_i_1);
                sp2 = _mm256_add_pd(sp2, diff_i_2);
                sp3 = _mm256_add_pd(sp3, diff_i_3);
                //[Opimize Before] sp_gt[t][sv_i] = sp_gt[t][sv_i] + diff[p_i][sv_i + 1];
            }
            sp0 = _mm256_add_pd(sp0, sp2);
            sp1 = _mm256_add_pd(sp1, sp3);
            sp = _mm256_add_pd(sp0, sp1);
            sp = _mm256_mul_pd(sp, div_vector);
            _mm256_storeu_pd(sp_start, sp);
        }

    }

    double sum_sv;
    for (int i = 0; i < train_size; i++)
    {
        sum_sv = 0.0;
        for (int j = 0; j < n_perm; j++) {
            // printf("%d, %d, %f\n", j, i, sp_gt[j][i]);
            sum_sv += sp_gt[j][i];
        }
        sv[i] = sum_sv / n_perm;
    }

    return sv;
}

/*
    v5: Computation Formula Optimization - Distance Computation
*/

// Optimized Version-Improved MC Approach
double* imc_compute_sv_optimize_v5(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length) {
    double** diff = (double**)malloc(sizeof(double*) * test_size);
    double* sv = (double*)malloc(sizeof(double) * train_size);
    double** sp_gt = (double**)malloc(sizeof(double*) * n_perm);
    int* x_trn_idx = (int*)malloc(sizeof(double) * (train_size + 1));
    double dis;
    for (int i = 0; i < n_perm; i++)
    {
        sp_gt[i] = (double*)malloc(sizeof(double) * train_size);
    }
    bool heap_changed = false;
    for (int i = 0; i < train_size; i++)
    {
        sv[i] = 0.0;
        x_trn_idx[i] = i + 1;
    }
    for (int t = 0; t < test_size; t++)
    {
        diff[t] = (double*)malloc(sizeof(double) * (train_size + 1));
        diff[t][0] = 0.0;
    }


    double** dis_arr = (double**)malloc(sizeof(double*) * test_size);
    for (int t = 0; t < test_size; t++)
    {
        dis_arr[t] = (double*)malloc(sizeof(double) * train_size);
    }

    double div_test_size = 1.0 / (double)test_size;
    double K_divider = 1.0 / (double)K;
    double min_divider;
    __m256d div_vector = _mm256_set_pd(div_test_size, div_test_size, div_test_size, div_test_size);


    double* dis_arr_trn;
    dis_arr_trn = (double*)malloc(sizeof(double) * train_size);


    // __m256d neg2 = _mm256_set1_pd(-2.0);
    __m256d const_coef = _mm256_set1_pd(0.5);

    for (int i_trn = 0; i_trn < train_size; i_trn += 4)
    {
        __m256d distance1 = _mm256_setzero_pd();
        __m256d distance2 = _mm256_setzero_pd();
        __m256d distance3 = _mm256_setzero_pd();
        __m256d distance4 = _mm256_setzero_pd();
        __m256d distance1_2 = _mm256_setzero_pd();
        __m256d distance2_2 = _mm256_setzero_pd();
        __m256d distance3_2 = _mm256_setzero_pd();
        __m256d distance4_2 = _mm256_setzero_pd();
        for (int j = 0; j < feature_length; j += 8)
        {
            __m256d x_trn_1 = _mm256_loadu_pd(x_trn[i_trn] + j);
            __m256d x_trn_1_2 = _mm256_loadu_pd(x_trn[i_trn] + j + 4);
            __m256d x_trn_2 = _mm256_loadu_pd(x_trn[i_trn + 1] + j);
            __m256d x_trn_2_2 = _mm256_loadu_pd(x_trn[i_trn + 1] + j + 4);
            __m256d x_trn_3 = _mm256_loadu_pd(x_trn[i_trn + 2] + j);
            __m256d x_trn_3_2 = _mm256_loadu_pd(x_trn[i_trn + 2] + j + 4);
            __m256d x_trn_4 = _mm256_loadu_pd(x_trn[i_trn + 3] + j);
            __m256d x_trn_4_2 = _mm256_loadu_pd(x_trn[i_trn + 3] + j + 4);


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
        double* dis_start = dis_arr[i];
        for (int i_trn = 0; i_trn < train_size; i_trn += 4) {
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
            double* x_trn_start_2 = x_trn[i_trn + 1];
            double* x_trn_start_3 = x_trn[i_trn + 2];
            double* x_trn_start_4 = x_trn[i_trn + 3];

            __m256d trn_const = _mm256_loadu_pd(dis_arr_trn + i_trn);

            for (int j = 0; j < feature_length; j += 8) {
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
            // result = _mm256_mul_pd(neg2, result);
            result = _mm256_sub_pd(trn_const, result);
            _mm256_storeu_pd(dis_start + i_trn, result);
        }
    }


    for (int t = 0; t < n_perm; t++) {
        if (if_permute != 1) {
            random_shuffle_op(x_trn_idx, train_size);
        }
        for (int m = 0; m < test_size; m++) {
            // generate random permutation
            double last_v_sum = 0;
            double v_sum = 0;
            // initialize  max heap
            // HeapNode* heap = create_heap(K); 
            // [Optimize ]Heap Inlining
            double* heapArr = (double*)malloc(4 * K * sizeof(double));
            int* heapIdx = (int*)malloc(4 * K * sizeof(int));
            int heapSize = K;
            int heapcnt = 0;

            for (int i = 0; i < train_size; i += 1) {
                dis = dis_arr[m][i];
                // insert
                // [Optimize ]Heap Inlining
                // heap_changed = add_heap(heap, dis, x_trn_idx[i] - 1);
                double node = dis;
                int idx = x_trn_idx[i] - 1;
                if (heapcnt < heapSize - 1) {
                    heapArr[heapcnt] = node;
                    heapIdx[heapcnt] = idx;
                    heapcnt = heapcnt + 1;
                    // if heap changed true
                    // int n = ((i + 1) < K) ? i + 1 : K;
                    int n;
                    if (i > K) {
                        min_divider = K_divider;
                        n = K;
                    }
                    else {
                        min_divider = 1.0 / (double)(i + 1);
                        n = i + 1;
                    }

                    double v = 0.0;
                    for (int k = 0; k < n; k++) {
                        int ind = heapIdx[k];
                        if (y_trn[ind] == y_tst[m]) {
                            v += 1.0;
                        }
                    }
                    v_sum = v * min_divider;
                    diff[m][idx + 1] = v_sum - last_v_sum;
                    last_v_sum = v_sum;
                }
                else if (heapcnt == heapSize - 1) {
                    heapArr[heapcnt] = node;
                    heapIdx[heapcnt] = idx;
                    int root = 0;
                    int adjust_flag = 1;
                    while (adjust_flag == 1) {
                        int left = 2 * root + 1;
                        int right = 2 * root + 2;
                        if (left >= heapSize) {
                            adjust_flag = 0;
                            break;
                        }
                        int max = left;
                        if (right < heapSize && (heapArr[right] > heapArr[left]))
                            max = right;
                        if (heapArr[root] >= heapArr[max]) {
                            adjust_flag = 0;
                            break;
                        }
                        double temp = heapArr[root];
                        heapArr[root] = heapArr[max];
                        heapArr[max] = temp;
                        int tempidx = heapIdx[root];
                        heapIdx[root] = heapIdx[max];
                        heapIdx[max] = tempidx;
                        root = max;
                    }
                    // if heap changed true
                    heapcnt = heapcnt + 1;
                    // int n = ((i + 1) < K) ? i + 1 : K;
                    int n;
                    if (i > K) {
                        min_divider = K_divider;
                        n = K;
                    }
                    else {
                        min_divider = 1.0 / (double)(i + 1);
                        n = i + 1;
                    }
                    double v = 0.0;
                    for (int k = 0; k < n; k++) {
                        int ind = heapIdx[k];
                        if (y_trn[ind] == y_tst[m]) {
                            v += 1.0;
                        }
                    }
                    v_sum = v * min_divider;
                    diff[m][idx + 1] = v_sum - last_v_sum;
                    last_v_sum = v_sum;
                }
                else if (node > heapArr[0]) {
                    diff[m][idx + 1] = 0;
                }
                else {
                    heapArr[0] = node;
                    heapIdx[0] = idx;
                    int root = 0;
                    int adjust_flag = 1;
                    while (adjust_flag == 1) {
                        int left = 2 * root + 1;
                        int right = 2 * root + 2;
                        if (left >= heapSize) {
                            adjust_flag = 0;
                            break;
                        }
                        int max = left;
                        if (right < heapSize && (heapArr[right] > heapArr[left]))
                            max = right;
                        if (heapArr[root] >= heapArr[max]) {
                            adjust_flag = 0;
                            break;
                        }
                        double temp = heapArr[root];
                        heapArr[root] = heapArr[max];
                        heapArr[max] = temp;
                        int tempidx = heapIdx[root];
                        heapIdx[root] = heapIdx[max];
                        heapIdx[max] = tempidx;
                        root = max;
                    }
                    // if heap changed true
                    // int n = ((i + 1) < K) ? i + 1 : K;
                    int n;
                    if (i > K) {
                        min_divider = K_divider;
                        n = K;
                    }
                    else {
                        min_divider = 1.0 / (double)(i + 1);
                        n = i + 1;
                    }
                    double v = 0.0;
                    for (int k = 0; k < n; k++) {
                        int ind = heapIdx[k];
                        if (y_trn[ind] == y_tst[m]) {
                            v += 1.0;
                        }
                    }
                    v_sum = v * min_divider;
                    diff[m][idx + 1] = v_sum - last_v_sum;
                    last_v_sum = v_sum;
                }
            }

            free(heapArr);
            free(heapIdx);

        }


        for (int sv_i = 0; sv_i < train_size; sv_i += 4) {
            double* sp_start = sp_gt[t] + sv_i;
            __m256d sp0 = _mm256_setzero_pd();
            __m256d sp1 = _mm256_setzero_pd();
            __m256d sp2 = _mm256_setzero_pd();
            __m256d sp3 = _mm256_setzero_pd();
            __m256d sp = _mm256_setzero_pd();
            for (int p_i = 0; p_i < test_size; p_i += 4) {
                // printf("%d, %d, %f\n", p_i, sv_i, diff[p_i][sv_i]);
                __m256d diff_i_0 = _mm256_loadu_pd(diff[p_i] + sv_i + 1);
                __m256d diff_i_1 = _mm256_loadu_pd(diff[p_i + 1] + sv_i + 1);
                __m256d diff_i_2 = _mm256_loadu_pd(diff[p_i + 2] + sv_i + 1);
                __m256d diff_i_3 = _mm256_loadu_pd(diff[p_i + 3] + sv_i + 1);
                sp0 = _mm256_add_pd(sp0, diff_i_0);
                sp1 = _mm256_add_pd(sp1, diff_i_1);
                sp2 = _mm256_add_pd(sp2, diff_i_2);
                sp3 = _mm256_add_pd(sp3, diff_i_3);
                //[Opimize Before] sp_gt[t][sv_i] = sp_gt[t][sv_i] + diff[p_i][sv_i + 1];
            }
            sp0 = _mm256_add_pd(sp0, sp2);
            sp1 = _mm256_add_pd(sp1, sp3);
            sp = _mm256_add_pd(sp0, sp1);
            sp = _mm256_mul_pd(sp, div_vector);
            _mm256_storeu_pd(sp_start, sp);
        }

    }

    double sum_sv;
    for (int i = 0; i < train_size; i++)
    {
        sum_sv = 0.0;
        for (int j = 0; j < n_perm; j++) {
            // printf("%d, %d, %f\n", j, i, sp_gt[j][i]);
            sum_sv += sp_gt[j][i];
        }
        sv[i] = sum_sv / n_perm;
    }

    return sv;
}

double* imc_compute_sv(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length) {
    return imc_compute_sv_basic(x_trn, y_trn, x_tst, y_tst, K, train_size, test_size, n_perm, if_permute, feature_length);
}



double* imc_compute_sv_optimize(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length) {
    return imc_compute_sv_optimize_v5(x_trn, y_trn, x_tst, y_tst, K, train_size, test_size, n_perm, if_permute, feature_length);
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

void heap_is_valid(HeapNode* this) {
    assert(this != NULL);

    int start_ind = 0;

    while (start_ind < this->Size) {
        int left_child = 2 * start_ind + 1;
        int right_child = 2 * start_ind + 2;

        if (left_child < this->Size) {
            assert(this->Arr[start_ind] >= this->Arr[left_child]);
        }

        if (right_child < this->Size) {
            assert(this->Arr[start_ind] >= this->Arr[right_child]);
        }

        start_ind++;
    }
}

void test_heap() {
    HeapNode* heap = create_heap(5);
    assert(heap->Size == 5);

    // 1 2 7 8 9
    bool add = add_heap(heap, 2.0, 0);
    assert(add == true);

    add = add_heap(heap, 8.0, 1);
    assert(add == true);

    add = add_heap(heap, 9.0, 2);
    assert(add == true);

    add = add_heap(heap, 1.0, 3);
    assert(add == true);

    add = add_heap(heap, 7.0, 4);
    assert(add == true);
    print_heap(heap);

    heap_is_valid(heap);

    double max = heap->Arr[0];
    for (int i = 5; i < 100; i++) {
        double new_node = (double)rand() / RAND_MAX * 5;
        if (new_node > max) {
            assert(add_heap(heap, new_node, i) == false);
        }
        else {
            assert(add_heap(heap, new_node, i) == true);
        }
        max = heap->Arr[0];
        heap_is_valid(heap);
    }
    print_heap(heap);

    printf("Heap tests passed.\n");

}

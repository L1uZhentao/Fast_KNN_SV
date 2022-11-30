#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "include/tsc_x86.h"
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include "algorithm2.h"

#define CYCLES_REQUIRED 1e10
#define DATA_SIZE 10000
#define NUM_FEATURE 3072
#define TEST_SIZE 100
#define VAL_SIZE 1000
#define CALIBRATE true


bool compare_with_alg1_result(double* alg1, double* alg2, int arr_size, int print_debug_message) {
    double accumulate_diff = 0.0f;
    for (int i = 0; i < arr_size; i++) {
        if (print_debug_message == 1 && (alg1[i] != 0 || alg1[2] != 0))
        {
            printf("%d, alg1:%f, alg2:%f, diff:%f, diff / alg1[i] = %f\n", i, alg1[i], alg2[i], (alg1[i] - alg2[i]), (alg1[i] - alg2[i]) / alg1[i]);
        }
        accumulate_diff += (alg1[i] - alg2[i]) * (alg1[i] - alg2[i]);
    }
    if (print_debug_message == 1)
    {
        printf("%lf\n", sqrt(accumulate_diff));
    }
    return true;
}

// Returns the number of cycles needed to execute f
double measure_runtime_func(double* (*f)(double**, double*, double**, double*, int, int, int, int, int, int), 
              double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, 
              int test_size, int n_perm, int if_permute, int feature_length, double size_ratio)

{
    int** test_knn_gt;
    double cycles;
    myInt64 start;
    myInt64 start_knn;
    myInt64 start_sp;
    int i; int num_runs = 1;

    #ifdef CALIBRATE
        while (num_runs < (1 << 14)) {
            start = start_tsc();
            for (i = 0; i < num_runs; ++i)
            {
                f(x_trn, y_trn, x_tst, y_tst, K, train_size, test_size, n_perm, if_permute, feature_length);
            }
            cycles = stop_tsc(start);
            if (cycles >= CYCLES_REQUIRED) break;
            num_runs *= 2;
        }
    #endif

    if (num_runs < 10) num_runs = 20;
        printf("number of iterations, %d\n", num_runs);
        start = start_tsc();
        for (int j = 0; j < num_runs; j++)
        {
            f(x_trn, y_trn, x_tst, y_tst, K, train_size, test_size, n_perm, if_permute, feature_length);
        }
    cycles = (double)stop_tsc(start) / num_runs;
    printf("test_size: %d, size_ratio : %f\n", test_size, size_ratio);
    printf("Current version needed\t%lf\tcycles in total\n",  cycles);
    return cycles;
}

int main(int argc, char* argv[])
{
    // test_heap();
    if (argc < 4)
    {
        printf("usage: ./algorithm2 [training data batch] [testing data batch] [TEST SIZE] [-t/-s/-h] [feature size] [-d (optional-)]\n");
        exit(0);
    }

    int test_correctness = 0;
    int test_speedup = 0;
    int print_debug_message = 0;
    int feature_num = NUM_FEATURE;

    if (argc >= 5 && strcmp(argv[4], "-s") == 0) {
        test_speedup = 1;
    }

    if (argc >= 5 && strcmp(argv[4], "-t") == 0) {
        test_correctness = 1;
    }

    if (argc >= 5 && strcmp(argv[4], "-h") == 0) {
        srand(time(0));
        test_heap();
        exit(0);
    }

    if (argc >= 6)
    {
        int temp_feature_size = atoi(argv[5]);
        if (temp_feature_size > 0 && temp_feature_size < 3072)
            feature_num = temp_feature_size;
        // printf("%d", feature_num);
    }

    if ((argc >= 7 && strcmp(argv[6], "-d") == 0) || (argc >= 6 && strcmp(argv[5], "-d") == 0))
    {
        print_debug_message = 1;
    }

    FILE* file;
    FILE* file2;

    /*
     * Recall that in python implementation (exact_sp_example.py), we get x_trn from combining
     * data['features_training'] and data['features_testing']. As we create
     * 'CIFAR10_resnet50-keras_features.npz' on our own, data['features_training'] is actually
     * the features of data_batch_1, and data['features_testing'] being the features of test_batch.
     * (consideing the case that we get CIFAR10_resnet50-keras_features.npz by running
     * python3 unpack.py data_batch_1 test_batch.) Then in order to ensure the same result from both
     * python and c, please also run ./readData data_batch_1 test_batch. i.e. use the same data batches
     * as python.
     *
     * By running the above command, "data" will be a 2d array same as x_trn in python implementation,
     * "labels" will be a 2d array same as y_trn in python implementation. But we don't do shuffle as python
     * implementation do, so please comment out shuffle line in python if needing to test if 2 implementations
     * get same results.
     */

    unsigned char* labels = (unsigned char*)malloc(DATA_SIZE * 2 * sizeof(unsigned char));
    unsigned char** data = (unsigned char**)malloc(DATA_SIZE * 2 * sizeof(unsigned char*));

    file = fopen(argv[1], "rb");
    file2 = fopen(argv[2], "rb");

    int test_size = atoi(argv[3]);
    double size_ratio = (double)test_size / (double)TEST_SIZE;
    // printf("%f", size_ratio);
    if (size_ratio > 1.0) size_ratio = 1.0;

    for (int i = 0; i < DATA_SIZE; i++)
    {
        data[i] = (unsigned char*)malloc(NUM_FEATURE * sizeof(unsigned char));
        fread(&labels[i], 1, 1, file);
        fread(data[i], 1, NUM_FEATURE, file);
    }

    for (int i = DATA_SIZE; i < 2 * DATA_SIZE; i++)
    {
        data[i] = (unsigned char*)malloc(NUM_FEATURE * sizeof(unsigned char));
        fread(&labels[i], 1, 1, file2);
        fread(data[i], 1, NUM_FEATURE, file2);
    }

    double* labels_double = (double*)malloc(DATA_SIZE * 2 * sizeof(double));
    double** data_double = (double**)malloc(DATA_SIZE * 2 * sizeof(double*));

    int N_train = (int)(2 * DATA_SIZE - TEST_SIZE - VAL_SIZE) * size_ratio;
    int N_test = (int)TEST_SIZE * size_ratio;
    int N_val = (int)VAL_SIZE * size_ratio;
    printf("N_train:%d,N_test:%d,feature_num:%d\n", N_train, N_test, feature_num);
    for (int i = 0; i < 2 * DATA_SIZE; i++)
    {
        data_double[i] = (double*)malloc(feature_num * sizeof(double));
        labels_double[i] = labels[i];
        for (int j = 0; j < feature_num; j++)
            data_double[i][j] = data[i][j];
    }

    if (test_correctness) {
        printf("++");
        double* sv_alg_basic;
        double* sv_alg_optimize;
        printf("Computing SP values\n");
        sv_alg_basic = imc_compute_sv(data_double + (TEST_SIZE + VAL_SIZE), labels_double + (TEST_SIZE + VAL_SIZE), data_double, labels_double, 1, N_train, N_test, 1, 1, feature_num);
        sv_alg_optimize = imc_compute_sv_optimize(data_double + (TEST_SIZE + VAL_SIZE), labels_double + (TEST_SIZE + VAL_SIZE), data_double, labels_double, 1, N_train, N_test, 1, 1, feature_num);
        printf("Validating results\n");
        compare_with_alg1_result(sv_alg_basic, sv_alg_optimize, N_train, print_debug_message);
    } else if (test_speedup) {
        // double* imc_compute_sv_optimize_v1(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length);
        // imc_compute_sv(data_double + (TEST_SIZE + VAL_SIZE), labels_double + (TEST_SIZE + VAL_SIZE), data_double, labels_double, 1, N_train, N_test, 100, 1, feature_num);
        double** x_trn = data_double + (TEST_SIZE + VAL_SIZE);
        double* y_trn = labels_double + (TEST_SIZE + VAL_SIZE);
        double** x_tst = data_double;
        double* y_tst = labels_double;
        int K = 1;
        int n_perm = 100;
        int if_permute = 1;

        printf("Baisc version:\n");
        double cycles_basic = measure_runtime_func(imc_compute_sv_basic, x_trn, y_trn, x_tst, y_tst, K, N_train, N_test, n_perm, if_permute, feature_num, size_ratio);
        
        printf("Optimization 1: \n");
        double cycles_opt1 = measure_runtime_func(imc_compute_sv_optimize_v1, x_trn, y_trn, x_tst, y_tst, K, N_train, N_test, n_perm, if_permute, feature_num, size_ratio);
        printf("Speedup: %f\n", cycles_basic / cycles_opt1);

        printf("Optimization 2: \n");
        double cycles_opt2 = measure_runtime_func(imc_compute_sv_optimize_v2, x_trn, y_trn, x_tst, y_tst, K, N_train, N_test, n_perm, if_permute, feature_num, size_ratio);
        printf("Speedup: %f\n", cycles_basic / cycles_opt2);

        printf("Optimization 3: \n");
        double cycles_opt3 = measure_runtime_func(imc_compute_sv_optimize_v3, x_trn, y_trn, x_tst, y_tst, K, N_train, N_test, n_perm, if_permute, feature_num, size_ratio);
        printf("Speedup: %f\n", cycles_basic / cycles_opt3);

        printf("Optimization 4: \n");
        double cycles_opt4 = measure_runtime_func(imc_compute_sv_optimize_v4, x_trn, y_trn, x_tst, y_tst, K, N_train, N_test, n_perm, if_permute, feature_num, size_ratio);
        printf("Speedup: %f\n", cycles_basic / cycles_opt4);

        printf("Optimization 5: \n");
        double cycles_opt5 = measure_runtime_func(imc_compute_sv_optimize_v5, x_trn, y_trn, x_tst, y_tst, K, N_train, N_test, n_perm, if_permute, feature_num, size_ratio);
        printf("Speedup: %f\n", cycles_basic / cycles_opt5);

        printf("\nRuntime summary:");
        printf("%f * %f * %f * %f * %f * %f\n", 
                cycles_basic, cycles_opt1, cycles_opt2, cycles_opt3, cycles_opt4, cycles_opt5);
    } else {
        double cycles;
        myInt64 start;
        int i; int num_runs = 1;
#ifdef CALIBRATE
        while (num_runs < (1 << 14)) {
            start = start_tsc();
            for (i = 0; i < num_runs; ++i)
            {
                imc_compute_sv(data_double + (TEST_SIZE + VAL_SIZE), labels_double + (TEST_SIZE + VAL_SIZE), data_double, labels_double, 1, N_train, N_test, 10, 1, feature_num);
            }
            cycles = stop_tsc(start);
            if (cycles >= CYCLES_REQUIRED) break;
            num_runs *= 2;
        }
#endif
        printf("number of iterations, %d\n", num_runs);
        start = start_tsc();
        for (int j = 0; j < num_runs; j++)
        {
            imc_compute_sv(data_double + (TEST_SIZE + VAL_SIZE), labels_double + (TEST_SIZE + VAL_SIZE), data_double, labels_double, 1, N_train, N_test, 10, 1, feature_num);
            /*
            for (int k = 0; k < 20; k++) {
                printf("%d, %f\n", k, sv[0][k]);
            }
            for (int k = N_train - 20; k < N_train; k++) {
                printf("%d, %f\n", k, sv[0][k]);
            }
            */
        }
        cycles = (double)stop_tsc(start) / num_runs;
        printf("test_size:%d, size_ratio: %f, needed\t%lf\tcycles for testing data\n", test_size, size_ratio, cycles);

        free(labels);
        for (int j = 0; j < DATA_SIZE; j++)
        {
            free(data[j]);
        }
        free(data);

        fclose(file);
        fclose(file2);
    }
}
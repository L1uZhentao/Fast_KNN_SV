#include "include/tsc_x86.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include "algorithm1.h"

#define CYCLES_REQUIRED 1e10
#define DATA_SIZE 10000
#define TEST_SIZE 100
#define VAL_SIZE 1000
#define CALIBRATE false
#define TOTAL_FEATURE 3072

int is_equal(double a, double b, double tolerance) {
    return (fabs(a - b) <= tolerance * fabs(b));
}

void compare_with_python_result(double** c_result, int train_size, int test_size) {
    FILE *python_result;

    // You might need to update the path to this file
    char *filename = "python_result.txt";

    python_result = fopen(filename, "rb");
    if (python_result == NULL) {
        fprintf(stderr, "File %s not found. \n", filename);
    } 

    double** res = (double**)malloc(TEST_SIZE * sizeof(double*));
    for (int i = 0; i < TEST_SIZE; i++)
    {
        res[i] = (double*)malloc(train_size * sizeof(double));
        fread(res[i], 8, train_size, python_result);
    }

    int err_count = 0;

    for (int i = 0; i < test_size; i++)
    {
        for (int j = 0; j < train_size; j++) {
            if(!is_equal(c_result[i][j], res[i][j], 0.1)){
                printf("index: i %d j %d\n", i, j);
                printf("difference: %.17g\n", fabs(c_result[i][j] - res[i][j]));
                printf("c: %.17g python: %.17g\n", c_result[i][j], res[i][j]);
                err_count++;
            }
        }
    }

    printf("Error count: %d, total count:%d \n", err_count, test_size*train_size);

    for (int i = 0; i < test_size; i++) {
        free(res[i]);
    }
    free(res);
    fclose(python_result);
}

void compare_c_results(double** c_result1, double** c_result2) {
    int train_size = 2 * DATA_SIZE - TEST_SIZE - VAL_SIZE;
    for (int i = 0; i < TEST_SIZE; i++)
    {
        for (int j = 0; j < train_size; j++) {
            if(!is_equal(c_result1[i][j], c_result2[i][j], 0.1)){
                printf("index: i %d j %d\n", i, j);
                printf("difference: %.17g\n", fabs(c_result1[i][j] - c_result2[i][j]));
                printf("c: %.17g python: %.17g\n", c_result1[i][j], c_result2[i][j]);
                printf("error\n");
            }
        }
    }
}

// Returns the number of cycles needed to execute f
double measure_runtime_func(int** (*f)(double**, double**, int, int, int), 
              double** x_trn, double** x_tst, int train_size, int test_size, int feature_length,
              double* y_trn, double* y_tst, int K, double size_ratio)

{
    int** test_knn_gt;
    double cycles;
    double cycles_sp = 0;
    double cycles_knn = 0;
    myInt64 start;
    myInt64 start_knn;
    myInt64 start_sp;
    int i; int num_runs = 1;

    // #ifdef CALIBRATE
    //     while (num_runs < (1 << 14)) {
    //         start = start_tsc();
    //         for (i = 0; i < num_runs; ++i)
    //         {
    //             test_knn_gt = f(x_trn, x_tst, train_size, test_size, feature_length);
    //             compute_sp_optimize(x_trn, y_trn, test_knn_gt, y_tst, K, train_size, test_size);
    //         }
    //         cycles = stop_tsc(start);
    //         if (cycles >= CYCLES_REQUIRED) break;
    //         num_runs *= 2;
    //     }
    // #endif

    if (num_runs < 10) num_runs = 20;
        printf("number of iterations, %d\n", num_runs);
        start = start_tsc();
        for (int j = 0; j < num_runs; j++)
        {
            start_knn = start_tsc();
            test_knn_gt = f(x_trn, x_tst, train_size, test_size, feature_length);
            cycles_knn = cycles_knn + (double)stop_tsc(start_knn);
            start_sp = start_tsc();
            double** sp = compute_sp_optimize(x_trn, y_trn, test_knn_gt, y_tst, K, train_size, test_size);
            cycles_sp = cycles_sp + (double)stop_tsc(start_sp);
        }
    cycles = (double)stop_tsc(start) / num_runs;
    cycles_knn = cycles_knn / num_runs;
    cycles_sp = cycles_sp / num_runs;
    printf("test_size: %d, size_ratio : %f\n", test_size, size_ratio);
    printf("Current version needed\t%lf\tcycles in total, \t%lf\tcycles to compute knn, \t%lf\tcycles to compute sp.\n",  cycles, cycles_knn, cycles_sp);
    return cycles;
}


int main(int argc, char* argv[])
{
    

	if (argc < 4)
	{
		printf("usage: ./algorithm1 [training data batch] [testing data batch] [TEST SIZE] [-t/-s] [feature size]\n");
		exit(0);
	}

    int test_correctness = 0;
    int test_speedup = 0;
    int feature_num = TOTAL_FEATURE;

    if (argc >= 5 && strcmp(argv[4], "-s") == 0) {
        test_speedup = 1;
    }

    if (argc >= 5 && strcmp(argv[4], "-t") == 0) {
        test_correctness = 1;
    }

    if (argc >= 6)
    {
        int temp_feature_size = atoi(argv[5]);
        if (temp_feature_size > 0 && temp_feature_size < 3072)
        feature_num = temp_feature_size;
    }

    FILE *file;
    FILE *file2;

    /*
     * Recall that in python implementation (exact_sp_example.py), we get x_trn from combining
     * data['features_training'] and data['features_testing']. As we create
     * 'CIFAR10_resnet50-keras_features.npz' on our own, data['features_training'] is actually
     * the features of data_batch_1, and data['features_testing'] being the features of test_batch.
     * (considering the case that we get CIFAR10_resnet50-keras_features.npz by running
     * python3 unpack.py data_batch_1 test_batch.) Then in order to ensure the same result from both
     * python and c, please also run ./algorithm1 data_batch_1 test_batch. i.e. use the same data batches
     * as python.
     *
     * By running the above command, "data" will be a 2d array same as x_trn in python implementation,
     * "labels" will be a 2d array same as y_trn in python implementation. But we don't do shuffle as python
     * implementation does, so please comment out shuffle line in python if needing to test if 2 implementations
     * get same results.
     */

    unsigned char* labels = (unsigned char*) malloc (DATA_SIZE * 2 * sizeof(unsigned char));
    unsigned char** data = (unsigned char**) malloc (DATA_SIZE * 2 * sizeof(unsigned char*));

    file = fopen(argv[1], "rb");
    file2 = fopen(argv[2], "rb");

    if (file == NULL) {
    	fprintf(stderr, "File %s not found. \n", argv[1]);
    }

    if (file2 == NULL) {
    	fprintf(stderr, "File %s not found. \n", argv[2]);
    }

    int test_size = atoi(argv[3]);
    double size_ratio = (double) test_size / (double) TEST_SIZE;
    printf("%f\n", size_ratio);

    if (size_ratio > 1.0) size_ratio = 1.0;

    for (int i = 0; i < DATA_SIZE; i++)
    {
        data[i] = (unsigned char*)malloc(TOTAL_FEATURE * sizeof(unsigned char));
        fread(&labels[i], 1, 1, file);
        fread(data[i], 1, TOTAL_FEATURE, file);
    }

    for (int i = DATA_SIZE; i < 2 * DATA_SIZE; i++)
    {
        data[i] = (unsigned char*)malloc(TOTAL_FEATURE * sizeof(unsigned char));
        fread(&labels[i], 1, 1, file2);
        fread(data[i], 1, TOTAL_FEATURE, file2);
    }

    double* labels_double = (double*)malloc(DATA_SIZE * 2 * sizeof(double));
    double** data_double = (double**)malloc(DATA_SIZE * 2 * sizeof(double*));

    int N_train = (int) (2*DATA_SIZE - TEST_SIZE - VAL_SIZE) * size_ratio;
    int N_test = (int) TEST_SIZE * size_ratio;
    int N_val = (int) VAL_SIZE * size_ratio;
    printf("N_train:%d, N_test:%d\n", N_train, N_test);
    
    for (int i = 0; i < 2 * DATA_SIZE; i++)
    {
        data_double[i] = (double*)malloc(feature_num * sizeof(double));
        labels_double[i] = labels[i];
        for (int j = 0; j < feature_num; j++)
            data_double[i][j] = data[i][j];
    }

    /* This array is passed to x_tst_knn_gt function, and the function should write the output inside this array. */
    int** test_knn_gt = (int**)malloc(sizeof(int*) * TEST_SIZE);

    for (int i = 0; i < TEST_SIZE; i++)
    {
        test_knn_gt[i] = (int*)malloc(sizeof(int) * DATA_SIZE);
    }

    if (test_correctness) {
        printf("Computing SP values\n");
        test_knn_gt = get_true_KNN_optimize(data_double + (TEST_SIZE + VAL_SIZE), data_double, N_train, N_test, feature_num);
        double** knn = compute_sp_optimize(data_double + (TEST_SIZE + VAL_SIZE), labels_double + (TEST_SIZE + VAL_SIZE), test_knn_gt, labels_double, 1, N_train, N_test);
        printf("Validating results\n");
        compare_with_python_result(knn, N_train, N_test);
    }
    else if (test_speedup) {
        double** x_trn = data_double + (TEST_SIZE + VAL_SIZE);
        double** x_tst = data_double;
        double* y_trn = labels_double + (TEST_SIZE + VAL_SIZE);
        double* y_tst = labels_double;
        int K = 1;

        printf("test_size: % d, size_ratio : % f\n", test_size, size_ratio);

        printf("Baisc version:\n");
        double cycles_basic = measure_runtime_func(get_true_KNN_basic, x_trn, x_tst, N_train, N_test, feature_num, y_trn, y_tst, K, size_ratio);
        
        printf("Optimization 1: \n");
        double cycles_opt1 = measure_runtime_func(get_true_KNN_optimize1, x_trn, x_tst, N_train, N_test, feature_num, y_trn, y_tst, K, size_ratio);
        printf("Speedup: %f\n", cycles_basic / cycles_opt1);

        printf("Optimization 2: \n");
        double cycles_opt2 = measure_runtime_func(get_true_KNN_optimize2, x_trn, x_tst, N_train, N_test, feature_num, y_trn, y_tst, K, size_ratio);
        printf("Speedup: %f\n", cycles_basic / cycles_opt2);

        printf("Optimization 3: \n");
        double cycles_opt3 = measure_runtime_func(get_true_KNN_optimize3, x_trn, x_tst, N_train, N_test, feature_num, y_trn, y_tst, K, size_ratio);
        printf("Speedup: %f\n", cycles_basic / cycles_opt3);

        printf("Optimization 4: \n");
        double cycles_opt4 = measure_runtime_func(get_true_KNN_optimize4, x_trn, x_tst, N_train, N_test, feature_num, y_trn, y_tst, K, size_ratio);
        printf("Speedup: %f\n", cycles_basic / cycles_opt4);

        printf("Optimization 5: \n");
        double cycles_opt5 = measure_runtime_func(get_true_KNN_optimize5, x_trn, x_tst, N_train, N_test, feature_num, y_trn, y_tst, K, size_ratio);
        printf("Speedup: %f\n", cycles_basic / cycles_opt5);

        printf("Optimization 6: \n");
        double cycles_opt6 = measure_runtime_func(get_true_KNN_optimize6, x_trn, x_tst, N_train, N_test, feature_num, y_trn, y_tst, K, size_ratio);
        printf("Speedup: %f\n", cycles_basic / cycles_opt6);

        printf("Optimization 7: \n");
        double cycles_opt7 = measure_runtime_func(get_true_KNN_optimize7, x_trn, x_tst, N_train, N_test, feature_num, y_trn, y_tst, K, size_ratio);
        printf("Speedup: %f\n", cycles_basic / cycles_opt7);

        printf("Optimization 8: \n");
        double cycles_opt8 = measure_runtime_func(get_true_KNN_optimize8, x_trn, x_tst, N_train, N_test, feature_num, y_trn, y_tst, K, size_ratio);
        printf("Speedup: %f\n", cycles_basic / cycles_opt8);

        printf("\nRuntime summary:");
        printf("%f * %f * %f * %f * %f * %f * %f * %f * %f\n", 
                cycles_basic, cycles_opt1, cycles_opt2, cycles_opt3, cycles_opt4,
                cycles_opt5, cycles_opt6, cycles_opt7, cycles_opt8);

    } else {
        double cycles;
        myInt64 start;
        int i; int num_runs = 1;
#ifdef CALIBRATE
        while (num_runs < (1 << 14)) {
            start = start_tsc();
            for (i = 0; i < num_runs; ++i)
            {
                test_knn_gt = get_true_KNN_optimize8(data_double + (TEST_SIZE + VAL_SIZE), data_double, N_train, N_test, feature_num);
                compute_sp_optimize(data_double + (TEST_SIZE + VAL_SIZE), labels_double + (TEST_SIZE + VAL_SIZE), test_knn_gt, labels_double, 1, N_train, N_test);
                /*
                 * Recall that in python implementation, we separate the dataset as:
                 *
                 * x_tst, y_tst = x_trn[:100], y_trn[:100]
                 * x_val, y_val = x_trn[100:1100], y_trn[100:1100]
                 * x_trn, y_trn = x_trn[1100:], y_trn[1100:]
                 *
                 * tst contains 100 data, val contains 1100 data, trn contains the rest.
                 * we use TEST_SIZE, VAL_SIZE, DATA_SIZE respectively to controls the size in c.
                 * when passing the array to functions, we will pass
                 * (1) data pointer (2) start index (3) array size. See below get_true_KNN sample api
                 * for illustration. In python implementation, we simple use x_tst_knn_gt = get_true_KNN(x_trn, x_tst).
                 * However in C, we pass (data pointer, start index of x_trn, size of x_trn, start index of x_tst, size of x_tst, out array for x_tst_knn_gt)
                 */
                 /*** C version of get_true_KNN((uint8_t**)data, TEST_SIZE+VAL_SIZE, DATA_SIZE-TEST_SIZE-VAL_SIZE, 0, TEST_SIZE, test_knn_gt); ***/
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
            test_knn_gt = get_true_KNN_optimize(data_double + (TEST_SIZE + VAL_SIZE), data_double, N_train, N_test, feature_num);
            double** sv = compute_sp_optimize(data_double + (TEST_SIZE + VAL_SIZE), labels_double + (TEST_SIZE + VAL_SIZE), test_knn_gt, labels_double, 1, N_train, N_test);
        }
        cycles = (double)stop_tsc(start) / num_runs;
        printf("test_size:%d, size_ratio: %f, needed\t%lf\tcycles for testing data\n", test_size, size_ratio, cycles);
    }
    free(labels);
    for (int j = 0; j < DATA_SIZE; j++)
    {
        free(data[j]);
    }
    free(data);

    fclose(file);
    fclose(file2);
}
#define CYCLES_REQUIRED 1e10
#define DATA_SIZE 10000
#define NUM_FEATURE 3072
#define TEST_SIZE 100
#define VAL_SIZE 1000
#define CALIBRATE true

typedef struct Heap  HeapNode;
struct Heap {
    double* Arr;
    int* Idx;
    int Size;
    int cnt;
};

void print_heap(HeapNode* heap);
HeapNode* create_heap(int topk);
void adjust_heap(HeapNode* heap, int root);
bool add_heap(HeapNode* heap, double node, int idx);
void random_shuffle(double** a, int arr_size);
void random_shuffle_idx(int* a, int arr_size);
double get_distance_basic(double* a, double* b, int length);
double get_distance_optimize(double* a, double* b, int length);
double* imc_compute_sv(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length);
double* imc_compute_sv_basic(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length);
double* imc_compute_sv_optimize(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length);
double* imc_compute_sv_optimize_v1(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length);
double* imc_compute_sv_optimize_v2(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length);
double* imc_compute_sv_optimize_v3(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length);
double* imc_compute_sv_optimize_v4(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length);
double* imc_compute_sv_optimize_v5(double** x_trn, double* y_trn, double** x_tst, double* y_tst, int K, int train_size, int test_size, int n_perm, int if_permute, int feature_length);

void heap_is_valid(HeapNode* this);
void test_heap();
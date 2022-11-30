double get_distance_basic(double* a, double* b, int length);
double get_distance_optimize(double* a, double* b, int length);
double get_distance_optimize_vector(double* a, double* b, int length);

void sort_basic(double* a, int length, int* b);
void sort_optimize_merge_sort(double* a, int length, int* b);
void sort_optimize(double* a, int length, int* b);

int** get_true_KNN_basic(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length);
int** get_true_KNN_optimize(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length);
int** get_true_KNN_optimize1(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length);
int** get_true_KNN_optimize2(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length);
int** get_true_KNN_optimize3(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length);
int** get_true_KNN_optimize4(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length);
int** get_true_KNN_optimize5(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length);
int** get_true_KNN_optimize6(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length);
int** get_true_KNN_optimize7(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length);
int** get_true_KNN_optimize8(double** x_trn, double** x_tst, int train_size, int test_size, int feature_length);

double** compute_sp_basic(double** x_trn, double* y_trn, int** test_knn_gt, double* y_tst, int K, int train_size, int test_size);
double** compute_sp_optimize(double** x_trn, double* y_trn, int** test_knn_gt, double* y_tst, int K, int train_size, int test_size);
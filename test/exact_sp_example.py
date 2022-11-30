import time
import numpy as np
from sklearn.utils import shuffle
from exact_sp import get_true_KNN, compute_single_unweighted_knn_class_shapley
import sys 


'''
Usage: python3 exact_sp_examplt.py [TEST SIZE] [FEATURE SIZE]
'''

data = np.load('CIFAR10_resnet50-keras_features.npz')
test_size = int(sys.argv[1])
size_ratio = float(test_size/100)
train_size = int(18900*size_ratio)
print(train_size)

feature_size = 3072

if len(sys.argv) > 2:
	feature_size = int(sys.argv[2])
	if feature_size < 0 or feature_size > 3072:
		feature_size = 3072

# type: 2d arr of unit8
x_trn = np.vstack((data['features_training'], data['features_testing']))
y_trn = np.hstack((data['labels_training'], data['labels_testing']))

#x_trn, y_trn = shuffle(x_trn, y_trn, random_state=0)

x_trn = np.reshape(x_trn, (-1, 3072))
x_tst, y_tst = x_trn[:test_size, :feature_size], y_trn[:test_size]
x_val, y_val = x_trn[100:1100, :feature_size], y_trn[100:1100]
x_trn, y_trn = x_trn[1100:1100+train_size, :feature_size], y_trn[1100:1100+train_size]

# we are using 1-nn classifier
K = 1

x_tst_knn_gt = get_true_KNN(x_trn, x_tst)
#x_val_knn_gt = get_true_KNN(x_trn, x_val)

# type: 2d arr of float64
sp_gt = compute_single_unweighted_knn_class_shapley(x_trn, y_trn, x_tst_knn_gt, y_tst, K)
#val_sp_gt = compute_single_unweighted_knn_class_shapley(x_trn, y_trn, x_val_knn_gt, y_val, K)

sp_gt.tofile('python_result.txt')
# np.save('tst_exact_sp_gt_npy', sp_gt)
#np.save('val_exact_sp_gt', val_sp_gt)

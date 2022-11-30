import sys
import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    print(dictionary[b'data'][9999, :])
    # print(dictionary[b'labels'])
    # row1 = dictionary[b'data'][0, :]
    print(np.shape(dictionary[b'data']))
    return dictionary

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print("Usage: python3 unpack.py [data file name] [test data file name]")
		sys.exit(0)

	CIFAR10data = unpickle(sys.argv[1])
	features_training = np.array(CIFAR10data[b'data'][:, :])
	features_training = features_training.astype(float)
	labels_training = np.array(CIFAR10data[b'labels'][:])
	labels_training = labels_training.astype(float)
	original_stdout = sys.stdout

	'''
	with open('filename3.txt', 'a') as f:
		sys.stdout = f
		for x in range(features_training.shape[0]):
			for y in range(3072):
				print(str(features_training[x][y])+' ', file=f, end = '')
			print()
		sys.stdout = original_stdout
	'''
	

	CIFAR10testdata = unpickle(sys.argv[2])
	features_testing = np.array(CIFAR10testdata[b'data'][:, :])
	features_testing = features_testing.astype(float)
	labels_testing = np.array(CIFAR10testdata[b'labels'][:])
	labels_testing = labels_testing.astype(float)
	'''
	with open('filename.txt', 'a') as f:
		sys.stdout = f	
		for x in range(features_testing.shape[0]):
			for y in range(3072):
					print(str(features_testing[x][y])+' ', file=f, end = '')
			print()
		sys.stdout = original_stdout
	'''
	np.savez('CIFAR10_resnet50-keras_features.npz', features_training=features_training, labels_training=labels_training, features_testing=features_testing, labels_testing=labels_testing)
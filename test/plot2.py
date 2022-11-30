import os
import os
import numpy
import subprocess
import matplotlib.pyplot as plt
import numpy as np

feature_length = 3072
multiplier = 189

# Run C base implementation
def run_c_base(data_batch, test_batch, test_size):
	#os.system('gcc ../readData.c -o readData')
	cmd = 'algorithm2 ' + data_batch + ' ' + test_batch + ' ' + str(test_size)
	print(cmd)
	res = os.popen(cmd)
	res = res.read()
	print(res)
	return res

# Run C optimized implementation

if __name__ == '__main__':
	data_batch = '1'
	n_perms = 100
	print('Testing data batch ' + data_batch)
	# The number of rows taken from test data matrix
	test_size = [((i+1)*20) for i in range(5)] #+ [(i+1)*10 for i in range(10)]
	os.system('gcc ../algorithm2.c ../main2.c -o algorithm2 -lm -mavx2 -march=native')
	performances = []
	for s in test_size:
		res = run_c_base('cifar-10-batches-bin/data_batch_' + data_batch + '.bin', 'cifar-10-batches-bin/test_batch.bin ', s)
		cycle = float(res.split("\t")[1])
		train_size = s * multiplier
		flops = 4 * n_perms * feature_length * train_size * s + n_perms * train_size * (s + 1) + train_size * (n_perms + 1)
		performance = flops / cycle
		print("performance:", performance)
		performances.append(performance)


	plt.title(r'Performance Plot of Algorithm 1')
	plt.xlabel('Input Size')
	plt.ylabel('Flops/Cycle')
	plt.plot(test_size, performances)
	plt.show()
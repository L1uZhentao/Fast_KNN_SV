import os
import os
import numpy
import subprocess
import matplotlib.pyplot as plt
import numpy as np

feature_length = 3072

# Run C base implementation
def run_algo1(data_batch, test_batch, test_size):
	cmd = './algo1 ' + data_batch + ' ' + test_batch + ' ' + str(test_size) + ' -s'
	print('Rnning command: ' + cmd)
	res = os.popen(cmd).read()
	print(res)
	return res

def run_algo2(data_batch, test_batch, test_size):
	cmd = './algo2 ' + data_batch + ' ' + test_batch + ' ' + str(test_size) + ' -s'
	print('Rnning command: ' + cmd)
	res = os.popen(cmd).read()
	print(res)
	return res

# Run C optimized implementation
if __name__ == '__main__':
	algo_to_run = 1  # Change here if want a plot for algo2
	num_versions = 0
	data_batch = '1'
	print('Testing data batch ' + data_batch)

	# The number of rows taken from test data matrix
	# 20, 40, 60, 80, 100
	test_sizes = [((i+1)*20) for i in range(4)] #+ [(i+1)*10 for i in range(10)]
	
	# Compile source code
	if algo_to_run == 1:
		os.system('gcc-11 -mavx2 -march=native ../main1.c ../algorithm1.c -o algo1')
		num_versions = 9
	else:
		os.system('gcc-11 -mavx2 -march=native ../main2.c ../algorithm2.c -o algo2')
		num_versions = 6
	
	all_cycles = [[] for _ in range(num_versions)]
	for s in test_sizes:
		if algo_to_run == 1:
			res = run_algo1('cifar-10-batches-bin/data_batch_' + data_batch + '.bin', 'cifar-10-batches-bin/test_batch.bin ', s)
		else:
			res = run_algo2('cifar-10-batches-bin/data_batch_' + data_batch + '.bin', 'cifar-10-batches-bin/test_batch.bin ', s)
		
		res = res.split('Runtime summary:')[1]
		res = res.split(' * ')
		res = [res[i].strip() for i in range(len(res))]
		for i in range(num_versions):
			all_cycles[i].append(res[i])  
		print(all_cycles)

	plt.title(r'Algorithm ' + algo_to_run)
	plt.xlabel('Input Size')
	plt.ylabel('Flops/Cycle')
	for cycles in all_cycles:
		plt.plot(test_sizes, cycles)
	plt.show()
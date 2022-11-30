import os
import numpy
import subprocess
# Run original python implementation
def run_python_org(data_batch, test_batch, test_size):
	# Unpack data
	cmd = 'python unpack.py {data} {test}'.format(data=data_batch, test=test_batch)
	os.system(cmd)

	# Run with the unpacked data
	cmd = 'python exact_sp_example.py ' + str(test_size)
	os.system(cmd)

# Run C base implementation
def run_c_base(data_batch, test_batch, test_size):
	os.system('gcc -mavx2 -march=native ../main1.c ../algorithm1.c -o algo1')
	cmd = './algo1 ' + data_batch + ' ' + test_batch + ' ' + str(test_size) + ' -t'
	print(cmd)
	os.system(cmd)

# Run C optimized implementation

if __name__ == '__main__':
	data_batch = '1'
	test_size = 12
	print('Testing data batch ' + data_batch)
	print('Running Python program')
	run_python_org('cifar-10-batches-py/data_batch_' + data_batch, 'cifar-10-batches-py/test_batch', test_size)
	print('Running C program')
	run_c_base('cifar-10-batches-bin/data_batch_' + data_batch + '.bin', 'cifar-10-batches-bin/test_batch.bin', test_size)
	
	# print("testing data batch 2")
	# run_python_org("cifar-10-batches-py/data_batch_2", "cifar-10-batches-py/test_batch")
	# run_c_base('cifar-10-batches-bin/data_batch_2.bin', 'cifar-10-batches-bin/test_batch.bin')
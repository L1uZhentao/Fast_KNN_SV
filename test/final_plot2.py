import os
import os
import numpy
import subprocess
import matplotlib.pyplot as plt
import numpy as np


algo1_records = [[169595310.900000, 788897782.800000, 4022523020.600000, 25575049179.000000, 185759050697.700012],
[134954185.800000, 655567831.200000, 3479162613.000000, 23209560819.799999, 173886043149.100006], 
[110835758.200000, 461166607.900000, 1833604711.000000, 7508644302.300000, 29497125997.299999],
[79994709.700000, 332546000.800000, 1317324087.800000, 5364877787.000000, 21072127623.900002], 
[67653453.800000, 281822662.300000, 1125067816.500000, 4573517233.700000, 18156722323.599998],
[57841298.600000, 244638813.800000, 1025392892.100000, 3972325299.800000, 15659674118.799999], 
[146873395.000000, 730043357.800000, 4031928524.900000, 16287991312.900000, 77054772596.500000],
[49859458.200000, 210347920.400000, 855660993.200000, 3441320186.200000, 13518474811.799999],
[50331707.500000, 193298091.100000, 733610107.300000, 2900348866.100000, 11260366527.700001]]


algo2_records = [[14921049907.299999, 59557911793.900002, 237905489515.899994, 951538232026.500000, 3780752758389.100098], 
[5751384102.000000, 23174917106.599998, 92464932034.300003, 368775234001.599976, 1468147693648.399902], 
[5722552960.700000, 23158713971.700001, 92795508372.699997, 367344153576.200012, 1470662867506.500000],
[74992360.984375, 294590937.578125, 1153768720.562500, 4536705201.800000, 18230167622.799999],
[70522376.917969, 275085618.218750, 1094236511.375000, 4262355655.300000, 17152188904.200001],
[62782911.089844, 224615076.265625, 840769441.125000, 3217037982.200000, 12722406446.000000]]

test_size = [4, 8, 16, 32, 64]
train_size = []
feature = 3072
perm = 100

for x in range(len(test_size)):
	train_size.append(test_size[x]*189)

algo1_flopsb_7 = []
algo1_flops8 = []
algo2_flopsb = []
algo2_flops1_2 = []
algo2_flops3_4 = []
algo2_flops5 = []
for x in range(len(test_size)):
	algo1_flopsb_7.append(4*feature*test_size[x]*train_size[x]+test_size[x]+4*test_size[x]*(train_size[x]-1))
	algo1_flops8.append(train_size[x]*test_size[x]*(2*feature+9)+9*train_size[x]+2*train_size[x]*feature)
	algo2_flopsb.append(4*perm*feature*train_size[x]*test_size[x]+perm*train_size[x]*(test_size[x]+1)+train_size[x]*(perm+1))
	algo2_flops1_2.append(train_size[x]*(perm+1)+perm*train_size[x]*(test_size[x]+1)+test_size[x]*train_size[x]*(3*feature+8)*perm)
	algo2_flops3_4.append(train_size[x]*test_size[x]*(3*feature+8)+perm*train_size[x]*(test_size[x]+4)+train_size[x]*(perm+1))
	algo2_flops5.append(2*feature*train_size[x]+9*train_size[x]+perm*train_size[x]*(test_size[x]+4)+train_size[x]*(perm+1)+train_size[x]*test_size[x]*(2*feature+9))

print('algo1_flopsb_7',algo1_flopsb_7)
print('algo1_flops8',algo1_flops8)
print('algo2_flopsb',algo2_flopsb)
print('algo2_flops1_2',algo2_flops1_2)
print('algo2_flops3_4',algo2_flops3_4)
print('algo2_flops5',algo2_flops5)




algo1_label_mapping = {
	0: "basic implementation",
	1: "optimization 1",
	2: "optimization 2",
	5: "optimization 3",
	7: "optimization 4",
	8: "optimization 5"
}

algo2_label_mapping = {
	0: "basic implementation",
	1: "optimization 1",
	2: "optimization 2",
	3: "optimization 3",
	4: "optimization 4",
	5: "optimization 5"
}

'''
plt.title(r'Algorithm 1 Runtime Plot')
plt.xlabel('Training + Testing data Size (MB)')
plt.ylabel('Runtime (second)')

for x in range(len(algo1_records)):
	if x == 0 or x == 1 or x == 2 or x == 5 or x == 7 or x == 8:
		plt.plot([4, 8, 16, 32, 64], [c/1100000000 for c in algo1_records[x]], label=algo1_label_mapping[x])
plt.xticks([4, 8, 16, 32, 64], ['18.7', '37.4', '74.7', '149.4', '298.9'])
plt.tick_params(axis='x', which='major', labelsize=7)
plt.legend()
plt.savefig('algo1_runtime_plot.png')
'''

'''
plt.title(r'Algorithm 2 Runtime Plot')
plt.xlabel('Training + Testing data Size (MB)')
plt.ylabel('Runtime (second)')

for x in range(len(algo2_records)):
	plt.plot([4, 8, 16, 32, 64], [c/1100000000 for c in algo2_records[x]], label=algo2_label_mapping[x])
plt.xticks([4, 8, 16, 32, 64], ['18.7', '37.4', '74.7', '149.4', '298.9'])
plt.tick_params(axis='x', which='major', labelsize=7)
plt.legend()
plt.savefig('algo2_runtime_plot.png')
'''

'''
plt.title(r'Algorithm 1 Performance Plot')
plt.xlabel('Training + Testing data size (MB)')
plt.ylabel('Performance (flops/cycle)')

for x in range(len(algo1_records)):
	# performance_list = []
	if not(x == 0 or x == 1 or x == 2 or x == 5 or x == 7 or x == 8):
		continue
	if x <= 7:
		plt.plot([4, 8, 16, 32, 64], [(flop/cycle) for cycle, flop in zip(algo1_records[x], algo1_flopsb_7)], label=algo1_label_mapping[x])
	else:
		plt.plot([4, 8, 16, 32, 64], [(flop/cycle) for cycle, flop in zip(algo1_records[x], algo1_flops8)], label=algo1_label_mapping[x])
plt.xticks([4, 8, 16, 32, 64],  ['18.7', '37.4', '74.7', '149.4', '298.9'])
plt.tick_params(axis='x', which='major', labelsize=7)
plt.legend()
plt.savefig('algo1_performance_plot.png')
'''


'''
plt.title(r'Algorithm 2 Performance Plot')
plt.xlabel('Training + Testing data size (MB)')
plt.ylabel('Performance (flops/cycle)')

for x in range(len(algo2_records)):
	# performance_list = []
	if x == 0:
		plt.plot([4, 8, 16, 32, 64], [(flop/cycle) for cycle, flop in zip(algo2_records[x], algo2_flopsb)], label=algo2_label_mapping[x])
	elif x <= 2:
		plt.plot([4, 8, 16, 32, 64], [(flop/cycle) for cycle, flop in zip(algo2_records[x], algo2_flops1_2)], label=algo2_label_mapping[x])
	elif x <= 4:
		plt.plot([4, 8, 16, 32, 64], [(flop/cycle) for cycle, flop in zip(algo2_records[x], algo2_flops3_4)], label=algo2_label_mapping[x])
	else:
		plt.plot([4, 8, 16, 32, 64], [(flop/cycle) for cycle, flop in zip(algo2_records[x], algo2_flops5)], label=algo2_label_mapping[x])
plt.xticks([4, 8, 16, 32, 64], ['18.7', '37.4', '74.7', '149.4', '298.9'])
plt.tick_params(axis='x', which='major', labelsize=7)
plt.legend()
plt.savefig('algo2_performance_plot.png')
'''

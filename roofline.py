import matplotlib.pyplot as plt
import numpy as np
 
# Creating vectors X and Y
x = np.linspace(0, 16384, 100)
y = 41.6*x
 
fig = plt.figure(figsize = (10, 5))
test_size = 64
train_size = 64*189
feature_size = 3072
n_perm = 100
K = 5

### algo1
'''
flops = 4837734720
memory = 4 * test_size * train_size + 8 * train_size * 2 + 8 * train_size * 2 + 8 * train_size * 3072 + 8 * test_size * 3072 + 8 * train_size * test_size + train_size * 8 + test_size * 8
perf = 0.43

### algo1 basic
flopsb = 9515777856
memoryb = 4 * test_size * train_size + 8 * train_size * 2 + 8 * train_size * 2 + 8 * train_size * 3072 + 8 * test_size * 3072 + 8 * train_size * test_size + train_size * 8 + test_size * 8
perfb = 0.05
plt.plot(flops / memory, perf, marker = "o", label = "Algo 1 optimized")
plt.plot(flopsb / memoryb, perfb, marker = "x", label = "Algo 1 basic")
print(memory)
'''

### algo2

flops2 = 4921209216
memory2 = 8 * test_size * (train_size + 1) + 8 * train_size + 8 * n_perm * train_size + (train_size + 1) * 4 + 8 * test_size * train_size + 8 * train_size + 4 * K * 8 + 4 * K * 4 + train_size * feature_size + test_size * feature_size + train_size * 8 + test_size * 8
perf2 = 0.39

flops2b = 951347992896
memory2b = 8 * test_size * (train_size + 1) + 8 * train_size + 8 * n_perm * train_size + (train_size + 1) * 4 + 8 * test_size * train_size + 8 * train_size + 4 * K * 8 + 4 * K * 4 + train_size * feature_size + test_size * feature_size + train_size * 8 + test_size * 8
perf2b = 0.25
plt.plot(flops2 / memory2, perf2, marker = "o", label="Algo 2 optimized")
plt.plot(flops2b / memory2b, perf2b, marker = "x", label="Algo 2 basic")


# Create the plot
plt.plot(x, y, label = "Memory Bandwidth")
y1 = [16] * 100
y2 = [4] * 100
plt.plot(x, y1, "--", label="With SIMD")
plt.plot(x, y2, label="Without SIMD")
plt.xscale('log', basex=2)
plt.yscale('log', basey=2)


#plt.xticks([1/16, 1/8,1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512], [r'$\frac{1}{16}$', r'$\frac{1}{8}$',r'$\frac{1}{4}$',r'$\frac{1}{2}$',r'$1$', r'$2$', r'$4$', r'$8$', r'$16$', r'$32$', r'$64$', r'$128$', r'$256$', r'$512$'])
#plt.yticks([1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512], [r'$\frac{1}{4}$',r'$\frac{1}{2}$',r'$1$', r'$2$', r'$4$', r'$8$', r'$16$', r'$32$', r'$64$', r'$128$', r'$256$', r'$512$'])
plt.xticks([1/16, 1/8,1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], [r'$\frac{1}{16}$', r'$\frac{1}{8}$',r'$\frac{1}{4}$',r'$\frac{1}{2}$',r'$1$', r'$2$', r'$4$', r'$8$', r'$16$', r'$32$', r'$64$', r'$128$', r'$256$', r'$512$', r'$1024$', r'$2048$', r'$4096$', r'$8192$', r'$16384$'])
plt.yticks([1/4, 1/2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512], [r'$\frac{1}{4}$',r'$\frac{1}{2}$',r'$1$', r'$2$', r'$4$', r'$8$', r'$16$', r'$32$', r'$64$', r'$128$', r'$256$', r'$512$'])
plt.ylabel("P(n) flops / cycle")
plt.xlabel("I(n) flops / byte")
plt.ylim(0, 512)
plt.legend()
# Save the plot
plt.savefig("roofline_algo2.png")
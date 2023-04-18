import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import numpy as np
from scipy.ndimage import convolve

# Developed by Jamila Taaki
# Prototype CUDA kernel for efficient computation of a transit search over gridded candidate search space: y^T K t / sqrt(t K t^T)
# The output (for fixed d) is a matrix (N/2, N/2) of detection statistics, row, col = period, epoch
# Scatter scatterism (each task takes an element of the noise covariance and 'scatters' it into transit detection tests)

#=============================================
# Comparison CPU kernel (runs on a single core)
# Scatter pattern: iterates over elements of covariance
def CPU_transit_num_scatter(output_array, input_array, half_N):
    for i in range(len(input_array)):
        for p in range(1, half_N+1):
            output_array[p-1,i%p] += input_array[i]

def CPU_transit_den_scatter(output_array, input_array, half_N):
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            if i == j:
                for p in range(1, half_N+1):
                    output_array[p-1, i%p] += input_array[i, j]
            else:
                for p in range(1, np.min((np.abs(i-j)+1, half_N+1))):
                    if np.abs(i-j)%p == 0: output_array[p-1, i%p] += input_array[i, j]

# Gather pattern: iterates over candidate transit signals
def CPU_transit_num_gather(output_array, input_array, half_N):
    for p in range(1, half_N+1):
        for t0 in range(p): output_array[p-1, t0] = np.sum(input_array[t0::p])

def CPU_transit_den_gather(output_array, input_array, half_N):
    for p in range(1, half_N+1):
        for t0 in range(p): output_array[p-1, t0] = np.sum(input_array[t0::p, t0::p])

#=============================================
# CUDA kernel

start = drv.Event()
end = drv.Event()

#for gather, for the threads in a block possible take one transit but each read and sum (T_N/no threads) elements, then synch, then sum and write to output
# Naive gather kernel is fastest, approximate threads in a block share similar p

mod = SourceModule("""

__device__ void collision_free_write(float* output, int output_idx, float value)
{
    while (true)
        {
            float old_value = output[output_idx];
            if (__int_as_float(atomicCAS((int *) &output[output_idx], __float_as_int(old_value), __float_as_int(old_value+value))) == old_value )
            {
                 break;
            }
        }
}

__global__ void gather_transit_num(float *output, float *input, int half_N)
{
    int tx = blockIdx.x*blockDim.x + threadIdx.x; //t0
    int ty = blockIdx.y*blockDim.y + threadIdx.y;
    int p = ty + 1;
    int no_transit = floorf(((2*half_N)-tx-1)/p)+1;
    float value = 0.0;
    if (tx < p && p <= half_N) {
        for (int i = 0; i < no_transit; ++i) {
            value += input[(i*ty) + tx];
        }
        output[half_N*ty + tx] = value;
    }
}

__global__ void gather_transit_den(float *output, float *input, int half_N)
{ // Redundant (simple) version
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int ty = blockIdx.y*blockDim.y + threadIdx.y;
    int p = ty + 1;
    int t0 = tx;
    int no_transit = floorf(((2*half_N)-t0-1)/p)+1;
    float value = 0.0;
    if (t0 < p && p <= half_N) {
        for (int i=0; i < no_transit; ++i) {
            for (int j=0; j < no_transit; ++j) {
                value += input[((i*ty)+tx)*(2*half_N) + (j*ty) + tx];
            }
        }
        output[(half_N * (p-1)) + t0] = value;
    }
}


__global__ void scatter_transit_num(float *output, float *input, int half_N)
{   // Each thread takes an element of input y_d and value P, write to output index P*half_N + tx%p
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int ty = blockIdx.y*blockDim.y + threadIdx.y;
    int p = ty + 1; // period index
    float value = input[tx];
    int t0 = tx%p; // epoch
    if (p <= half_N && tx < 2*half_N) {
        int output_idx = (half_N * (p-1)) + t0;
        collision_free_write(output, output_idx, value);
    }
}

__global__ void scatter_transit_den(float *output, float *input, int half_N)
{
    // For element of input K_d[i,j] calculate output test indices, by finding period as factors of |i-j| and epoch as min(i,j)%P
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    float value = input[ty*(half_N*2) + tx];
    if (ty < (2*half_N) && tx < (2*half_N)) {
        if (tx == ty) {
            for (int i=1; i<=half_N; ++i){
                int output_idx = (half_N * (i-1)) + tx%i;
	        collision_free_write(output, output_idx, value);
            }
        }
        else {
            for(int i=1; i<=min(int(abs(tx - ty)), half_N); ++i) {
                if (int(abs(tx-ty))%i == 0) {
                    int output_idx = (half_N*(i-1)) + tx%i;
    		    collision_free_write(output, output_idx, value);
                }
            }
        }
    }
}
""")

#=============================================
# Transit search, modeled for 2-minute short-cadence data, at transit duration d, with orbital period range [1, 1+delta,.., N/2] and epoch range [1, 1+delta, ..., N/2]. Unphysical epoch values should be discarded. 
# New structure will loop through all sectors and add result to global arr. 
# Also want to include ofir stuff, finer resolution at greater periods / or registration?
# for periods greater than sec len 

N_full = 28*720  #no. of samples in lightcurve: here modelling 28-days of 2-minute short cadence data
d = 60 # transit duration: 2 hours
delta = 5  # step size for orbital period and epoch: 10 min

N = (N_full-d)//delta
half_N = int(N/2)

print (N, half_N)

y_sim = np.random.uniform(0,1, N_full).astype(np.float32) # simulated lightcurve: y
cov_inv = np.random.uniform(0, 1, (N_full, N_full)).astype(np.float32) # simulated inverse covariance matrix: K
y_cov_inv = cov_inv.dot(y_sim) # y^T K
transit_profile = np.ones(d)
transit_kernel = np.outer(transit_profile, transit_profile) # transit kernel denoted as k_d

#=============================================
# K_d: compute inverse covariance * k_d and downsampled (depending on delta) once per lightcurve, per transit duration
# y_d: compute y^TK * k_d and downsampled ""

K_d = convolve(cov_inv, transit_kernel)[int(d/2)-1:N_full-int(d/2)-1,int(d/2)-1:N_full-int(d/2)-1][::delta,::delta].astype(np.float32)
y_d = convolve(y_cov_inv, transit_profile)[int(d/2)-1:N_full-int(d/2)-1][::delta].astype(np.float32)

# Equivalent computation (faster if delta is large)
#K_d_manual = np.zeros((N, N))
#y_d_manual = np.zeros(N)
#for i in range(N):
#    y_d_manual[i] = np.sum(transit_profile*y_cov_inv[(i*delta): (i*delta) + d])
#    for j in range(N):
#        K_d_manual[i,j] = np.sum(transit_kernel*cov_inv[(i*delta):(i*delta) + d, (j*delta):(j*delta) + d])
#print (np.allclose(K_d, K_d_manual))
#print (np.allclose(y_d, y_d_manual))

#=============================================
# CUDA GATHER thread tiling

total_no_transits = int(half_N*(half_N + 1)/2)
blockwidth = 16
gridwidth = int(np.ceil(half_N/blockwidth))

output_num = np.zeros(half_N**2, dtype=np.float32)
output_den = np.zeros(half_N**2, dtype=np.float32)

start.record()
start.synchronize()

gather_transit_den = mod.get_function("gather_transit_den")
gather_transit_den(drv.Out(output_den), drv.In(K_d), np.int32(half_N), block=(blockwidth, blockwidth, 1), grid=(gridwidth,gridwidth,1))

end.record()
end.synchronize()

secs = start.time_till(end)*1e-3
print ('(GPU gather) Time to calculate denominator of tests: ', secs)
print ('(GPU gather) Consistency check (period = 1) should provide the sum of the covariance: ', np.round(output_den[0]/np.sum(K_d), 2))

start.record()
start.synchronize()

gather_transit_num = mod.get_function("gather_transit_num")
gather_transit_num(drv.Out(output_num), drv.In(y_d), np.int32(half_N), block=(blockwidth, blockwidth, 1), grid=(gridwidth,gridwidth))

end.record()
end.synchronize()

secs = start.time_till(end)*1e-3
print ('(GPU gather) Time to calculate numerator of tests: ', secs)
transit_stats_GPU = np.divide(output_num, np.sqrt(output_den), out=np.zeros_like(output_num), where=output_den!=0.).reshape((half_N, half_N))

#=============================================
# CUDA SCATTER thread tiling

blockwidth = 10
gridwidth = int(np.ceil(N/blockwidth))

output_num = np.zeros(half_N**2, dtype=np.float32)
output_den = np.zeros(half_N**2, dtype=np.float32)

#=============================================
# CUDA: Calculate the numerator of every transit detection test
# CUDA kernel "scatter_num" takes y_d as input and outputs the numerator of every detection test, defined over P in [1.. N/2], epoch in [1.. N/2]

start.record()
start.synchronize()

scatter_transit_num = mod.get_function("scatter_transit_num")
scatter_transit_num(drv.Out(output_num), drv.In(y_d), np.int32(half_N), block=(blockwidth,blockwidth,1), grid=(gridwidth, int(gridwidth/2), 1))

end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print ('(GPU scatter) Time to calculate numerator of tests: ', secs)
#=============================================
# CUDA: Calculate the denominator of every transit detection test

start.record()
start.synchronize()

scatter_transit_den = mod.get_function("scatter_transit_den")
scatter_transit_den(drv.Out(output_den), drv.In(K_d), np.int32(half_N), block=(blockwidth, blockwidth, 1), grid=(gridwidth,gridwidth))

end.record()
end.synchronize()

secs = start.time_till(end)*1e-3
print ('(GPU scatter) Time to calculate denominator of tests: ', secs)
print ('(GPU scatter) Consistency check (period = 1) should provide the sum of the covariance: ', np.round(output_den[0]/np.sum(K_d), 2))

#=============================================
# Output transit detection tests, indexed as results[P, t_0], for indices where t_0 > P results are invalid
transit_stats_GPU = np.divide(output_num, np.sqrt(output_den), out=np.zeros_like(output_num), where=output_den!=0.).reshape((half_N, half_N))
#=============================================

output_num = np.zeros((half_N, half_N), dtype=np.float32)
output_den = np.zeros((half_N, half_N), dtype=np.float32)

start.record() 
start.synchronize()

CPU_transit_num_scatter(output_num, y_d, half_N)

end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print ('(CPU) Time to calculate numerator of tests: ', secs)

#=============================================

start.record() 
start.synchronize()

CPU_transit_den_scatter(output_den, K_d, half_N)

end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print ('(CPU scatter) Time to calculate denominator of tests: ', secs)
print ('(CPU scatter) Consistency check (period = 1) should provide the sum of the covariance: ', np.round(output_den[0,0]/np.sum(K_d), 2))

#=============================================
# Output transit detection tests, indexed as results[P, t_0], for indices where t_0 > P results are invalid
transit_stats_CPU = np.divide(output_num, np.sqrt(output_den), out=np.zeros_like(output_num), where=output_den!=0.)

#=============================================
output_num = np.zeros((half_N, half_N), dtype=np.float32)
output_den = np.zeros((half_N, half_N), dtype=np.float32)
start.record()
start.synchronize()

CPU_transit_den_gather(output_den, K_d, half_N)

end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print ('(CPU gather) Time to calculate denominator of tests: ', secs)
print ('(CPU gather) Consistency check (period = 1) should provide the sum of the covariance: ', np.round(output_den[0,0]/np.sum(K_d), 2))

#=============================================

#print (output_num, output_den)
#print (np.allclose(transit_stats_GPU, transit_stats_CPU)) #for large N the outputs tend to deviate due to cumulative fp err.

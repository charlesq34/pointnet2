#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include <string>
#include <vector>
using namespace std;
float randomf(){
    return (rand()+0.5)/(RAND_MAX+1.0);
}
static double get_time(){
    timespec tp;
    clock_gettime(CLOCK_MONOTONIC,&tp);
    return tp.tv_sec+tp.tv_nsec*1e-9;
}

// input: k (1), distance matrix dist (b,m,n)
// output: idx (b,m,k), val (b,m,k)
__global__ void selection_sort_gpu(int b, int n, int m, int k, float *dist, int *idx, float *val) {
    int batch_index = blockIdx.x;
    dist+=m*n*batch_index;
    idx+=m*k*batch_index;
    val+=m*k*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = dist+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s; 
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // update idx and val
            idx[j*n+s] = min;
            val[j*n+s] = p_dist[min];
            // swap min-th and i-th element
            float tmp = p_dist[min];
            p_dist[min] = p_dist[s];
            p_dist[s] = tmp;
        }
    }
}

int main()
{
    //int b=32,n=10000,m=1000,k=128;
    int b=32,n=2048,m=512,k=128;
    float *dist;
    int *idx;
    float *val;
    cudaMallocManaged(&dist, b*m*n*sizeof(float));
    cudaMallocManaged(&idx, b*m*k*sizeof(int));
    cudaMallocManaged(&val, b*m*k*sizeof(float));
    cudaMemset(idx, 0, sizeof(int)*b*m*k);
    for (int i=0;i<b*n*m;i++)
        dist[i]=randomf();

    double t0=get_time();
    selection_sort_gpu<<<b,256>>>(b,n,m,k,dist,idx,val);
    cudaDeviceSynchronize();
    printf("selection sort cpu time %f\n",get_time()-t0);

    return 0;
}

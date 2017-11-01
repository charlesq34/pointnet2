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
// output: idx (b,m,n), dist_out (b,m,n)
__global__ void selection_sort_gpu(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    int batch_index = blockIdx.x;
    dist+=m*n*batch_index;
    outi+=m*n*batch_index;
    out+=m*n*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s; 
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}

int main()
{
    //int b=32,n=10000,m=1000,k=128;
    int b=32,n=2048,m=512,k=128;
    //int b=2,n=4,m=2,k=3;
    float *dist;
    int *idx;
    float *dist_out;
    cudaMallocManaged(&dist, b*m*n*sizeof(float));
    cudaMallocManaged(&idx, b*m*n*sizeof(int));
    cudaMallocManaged(&dist_out, b*m*n*sizeof(float));
    cudaMemset(idx, 0, sizeof(int)*b*m*n);
    for (int i=0;i<b*n*m;i++)
        dist[i]=randomf();
    //for (int i=0;i<b*n*m;i++) {
    //    dist[i] = float(10-i);
    //    printf("%f ", dist[i]);
    //}
    //printf("\n");

    double t0=get_time();
    selection_sort_gpu<<<b,256>>>(b,n,m,k,dist,idx,dist_out);
    cudaDeviceSynchronize();
    printf("selection sort cpu time %f\n",get_time()-t0);
    
    //for (int i=0;i<b*n*m;++i)
    //    printf("%d ", idx[i]);
    //printf("\n");

    return 0;
}

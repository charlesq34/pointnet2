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
// output: idx (b,m,n), val (b,m,n)
void selection_sort_cpu(int b, int n, int m, int k, const float *dist, int *idx, float *val) {
    float *p_dist;
    float tmp;
    int tmpi;
    for (int i=0;i<b;++i) {
        for (int j=0;j<m;++j) {
            for (int s=0;s<n;++s) {
                val[i*m*n+j*n+s] = dist[i*m*n+j*n+s];
                idx[i*m*n+j*n+s] = s;
            }
        }
    }

    for (int i=0;i<b;++i) {
        for (int j=0;j<m;++j) {
            for (int s=0;s<n;++s)
                printf("%f ", dist[i*m*n+j*n+s]);
            printf("\n");
            p_dist = val+j*n;
            // selection sort for the first k elements
            for (int s=0;s<k;++s) {
                int min=s; 
                // find the min
                for (int t=s+1;t<n;++t) {
                    if (p_dist[t]<p_dist[min]) {
                        min = t;
                    }
                }
                printf("%d\n", min);
                // swap min-th and i-th element
                if (min!=s) {
                    tmp = p_dist[min];
                    p_dist[min] = p_dist[s];
                    p_dist[s] = tmp;
                    tmpi = idx[j*n+min];
                    idx[j*n+min] = idx[j*n+s];
                    idx[j*n+s] = tmpi;
                }       
            }
        }
        idx+=m*n;
        val+=m*n;
    }
}

int main()
{
    //int b=32,n=10000,m=1000,k=128;
    int b=2,n=4,m=2,k=3;
    float *dist=new float[b*m*n];
    int *idx=new int[b*m*n];
    float *val=new float[b*m*n];
    memset(idx, 0, sizeof(int)*b*m*n);
    //for (int i=0;i<b*n*m;i++)
    //    dist[i]=randomf();
    for (int i=0;i<b*n*m;i++) {
        dist[i] = float(10-i);
        printf("%f ", dist[i]);
    }
    printf("\n");



    double t0=get_time();
    selection_sort_cpu(b,n,m,k,dist,idx,val);
    printf("selection sort cpu time %f\n",get_time()-t0);

    for (int i=0;i<b*n*m;++i)
        printf("%d ", idx[i]);
    printf("\n");
    for (int i=0;i<b*n*m;++i)
        printf("%f ", val[i]);
    printf("\n");
    return 0;
}

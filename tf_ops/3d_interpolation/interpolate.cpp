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

// Find three nearest neigbors with square distance
// input: xyz1 (b,n,3), xyz2(b,m,3)
// output: dist (b,n,3), idx (b,n,3)
void threenn_cpu(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
     for (int i=0;i<b;++i) {
        for (int j=0;j<n;++j) {
	    float x1=xyz1[j*3+0];
	    float y1=xyz1[j*3+1];
	    float z1=xyz1[j*3+2];
            double best1=1e40; double best2=1e40; double best3=1e40;
            int besti1=0; int besti2=0; int besti3=0;
            for (int k=0;k<m;++k) {
                float x2=xyz2[k*3+0];
	        float y2=xyz2[k*3+1];
	        float z2=xyz2[k*3+2];
		//float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
		double d=x2*x2+y2*y2+z2*z2;
                if (d<best1) {
                    best3=best2;
                    besti3=besti2;
                    best2=best1;
                    besti2=besti1;
                    best1=d;
                    besti1=k;
                } else if (d<best2) {
                    best3=best2;
                    besti3=besti2;
                    best2=d;
                    besti2=k;
                } else if (d<best3) {
                    best3=d;
                    besti3=k;
                }
            } 
            dist[j*3]=best1;
            idx[j*3]=besti1;
            dist[j*3+1]=best2;
            idx[j*3+1]=besti2;
            dist[j*3+2]=best3;
            idx[j*3+2]=besti3;
        } 
        xyz1+=n*3;
        xyz2+=m*3;
        dist+=n*3;
        idx+=n*3;
    }
} 

// CONSTANT WEIGHT TODO
// input: dist (b,n,3)
// output: weight (b,n,3)
void get_weights_cpu(int b, int n, const float *dist, float *weight) {
    const float w = 1.0/3.0;
    for (int i=0;i<b;++i) {
        for (int j=0;j<n;++j) {
            weight[j*3]=w;
            weight[j*3+1]=w;
            weight[j*3+2]=w;
        } 
        dist+=n*3;
        weight+=n*3;
    }
}

// input: points (b,m,c), idx (b,n,3), weight (b,n,3)
// output: out (b,n,c)
void interpolate_cpu(int b, int m, int c, int n, const float *points, const int *idx, const float *weight, float *out) {
     float w1,w2,w3;
     int i1,i2,i3;
     for (int i=0;i<b;++i) {
        for (int j=0;j<n;++j) {
            w1=weight[j*3];
            w2=weight[j*3+1];
            w3=weight[j*3+2]; 
            i1=idx[j*3];
            i2=idx[j*3+1];
            i3=idx[j*3+2];
            for (int l=0;l<c;++l) {
                out[j*c+l] = points[i1*c+l]*w1 + points[i2*c+l]*w2 + points[i3*c+l]*w3;
            }
        } 
        points+=m*c;
        idx+=n*3;
        weight+=n*3;
        out+=n*c;
    }
}

// input: grad_out (b,n,c), idx (b,n,3), weight (b,n,3)
// output: grad_points (b,m,c)
void interpolate_grad_cpu(int b, int n, int c, int m, const float *grad_out, const int *idx, const float *weight, float *grad_points) {
     float w1,w2,w3;
     int i1,i2,i3;
     for (int i=0;i<b;++i) {
        for (int j=0;j<n;++j) {
            w1=weight[j*3];
            w2=weight[j*3+1];
            w3=weight[j*3+2]; 
            i1=idx[j*3];
            i2=idx[j*3+1];
            i3=idx[j*3+2];
            for (int l=0;l<c;++l) {
                grad_points[i1*c+l] += grad_out[j*c+l]*w1;
                grad_points[i2*c+l] += grad_out[j*c+l]*w2;
                grad_points[i3*c+l] += grad_out[j*c+l]*w3;
            }
        } 
        grad_out+=n*c;
        idx+=n*3;
        weight+=n*3;
        grad_points+=m*c;
    }
}

int main()
{
    int b=32,n=512,m=128,c=64;
    float *xyz1=new float[b*n*3];
    float *xyz2=new float[b*m*3];
    float *dist=new float[b*n*3];
    int *idx=new int[b*n*3];
    memset(idx, 0, sizeof(int)*b*n*3);
    float *weight=new float[b*n*3];
    float *points=new float[b*m*c];
    float *out=new float[b*n*c];
    float *grad_out=new float[b*n*c]; // grad to out
    memset(grad_out, 0.0, sizeof(float)*b*n*c);
    float *grad_points=new float[b*m*c]; // grad to points
    for (int i=0;i<b*n*3;i++)
        xyz1[i]=randomf();
    for (int i=0;i<b*m*3;i++)
        xyz2[i]=randomf();
    for (int i=0;i<b*m*c;i++)
        points[i]=randomf();

    double t0=get_time();
    threenn_cpu(b,n,m,xyz1,xyz2,dist,idx);
    printf("threenn cpu time %f\n",get_time()-t0);
    
    t0=get_time();
    get_weights_cpu(b,n,dist,weight);
    printf("get_weights_cpu cpu time %f\n",get_time()-t0);

    t0=get_time();
    interpolate_cpu(b,m,c,n,points,idx,weight,out);
    printf("interpolate_cpu cpu time %f\n",get_time()-t0);

    t0=get_time();
    interpolate_grad_cpu(b,n,c,m,grad_out,idx,weight,grad_points);
    printf("interpolate_grad_cpu cpu time %f\n",get_time()-t0);
    return 0;
}

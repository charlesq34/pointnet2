/* Furthest point sampling GPU implementation
 * Original author: Haoqiang Fan
 * Modified by Charles R. Qi
 * All Rights Reserved. 2017. 
 */

__global__ void cumsumKernel(int b,int n,const float * __restrict__ inp,float * __restrict__ out){
  const int BlockSize=2048;
  const int paddingLevel=5;
  __shared__ float buffer4[BlockSize*4];
  __shared__ float buffer[BlockSize+(BlockSize>>paddingLevel)];
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    float runningsum=0,runningsum2=0;
    for (int j=0;j<n;j+=BlockSize*4){
      int n24_i=min(n-j,BlockSize*4);
      int n24=(n24_i+3)&~3;
      int n2=n24>>2;
      for (int k=threadIdx.x*4;k<n24_i;k+=blockDim.x*4){
        if (k+3<n24_i){
          float v1=inp[i*n+j+k];
          float v2=inp[i*n+j+k+1];
          v2+=v1;
          float v3=inp[i*n+j+k+2];
          float v4=inp[i*n+j+k+3];
          v4+=v3;
          v3+=v2;
          v4+=v2;
          buffer4[k]=v1;
          buffer4[k+1]=v2;
          buffer4[k+2]=v3;
          buffer4[k+3]=v4;
          buffer[(k>>2)+(k>>(2+paddingLevel))]=v4;
        }else{
          float v=0;
          for (int k2=k;k2<n24_i;k2++){
            v+=inp[i*n+j+k2];
            buffer4[k2]=v;
          }
          for (int k2=n24_i;k2<n24;k2++){
            buffer4[k2]=v;
          }
          buffer[(k>>2)+(k>>(2+paddingLevel))]=v;
        }
      }
      int u=0;
      for (;(2<<u)<=n2;u++){
        __syncthreads();
        for (int k=threadIdx.x;k<int(n2>>(u+1));k+=blockDim.x){
          int i1=(((k<<1)+2)<<u)-1;
          int i2=(((k<<1)+1)<<u)-1;
          i1+=i1>>paddingLevel;
          i2+=i2>>paddingLevel;
          buffer[i1]+=buffer[i2];
        }
      }
      u--;
      for (;u>=0;u--){
        __syncthreads();
        for (int k=threadIdx.x;k<int((n2-(1<<u))>>(u+1));k+=blockDim.x){
          int i1=(((k<<1)+3)<<u)-1;
          int i2=(((k<<1)+2)<<u)-1;
          i1+=i1>>paddingLevel;
          i2+=i2>>paddingLevel;
          buffer[i1]+=buffer[i2];
        }
      }
      __syncthreads();
      for (int k=threadIdx.x*4;k<n24;k+=blockDim.x*4){
        if (k!=0){
          int k2=((k>>2)-1)+(((k>>2)-1)>>paddingLevel);
          buffer4[k]+=buffer[k2];
          buffer4[k+1]+=buffer[k2];
          buffer4[k+2]+=buffer[k2];
          buffer4[k+3]+=buffer[k2];
        }
      }
      __syncthreads();
      for (int k=threadIdx.x;k<n24_i;k+=blockDim.x){
        out[i*n+j+k]=buffer4[k]+runningsum;
      }
      float t=buffer[(n2-1)+((n2-1)>>paddingLevel)]+runningsum2;
      float r2=runningsum+t;
      runningsum2=t-(r2-runningsum);
      runningsum=r2;
      __syncthreads();
    }
  }
}

__global__ void binarysearchKernel(int b,int n,int m,const float * __restrict__ dataset,const float * __restrict__ query, int * __restrict__ result){
  int base=1;
  while (base<n)
    base<<=1;
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
      float q=query[i*m+j]*dataset[i*n+n-1];
      int r=n-1;
      for (int k=base;k>=1;k>>=1)
        if (r>=k && dataset[i*n+r-k]>=q)
          r-=k;
      result[i*m+j]=r;
    }
  }
}
__global__ void farthestpointsamplingKernel(int b,int n,int m,const float * __restrict__ dataset,float * __restrict__ temp,int * __restrict__ idxs){
  if (m<=0)
    return;
  const int BlockSize=512;
  __shared__ float dists[BlockSize];
  __shared__ int dists_i[BlockSize];
  const int BufferSize=3072;
  __shared__ float buf[BufferSize*3];
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    int old=0;
    if (threadIdx.x==0)
      idxs[i*m+0]=old;
    for (int j=threadIdx.x;j<n;j+=blockDim.x){
      temp[blockIdx.x*n+j]=1e38;
    }
    for (int j=threadIdx.x;j<min(BufferSize,n)*3;j+=blockDim.x){
      buf[j]=dataset[i*n*3+j];
    }
    __syncthreads();
    for (int j=1;j<m;j++){
      int besti=0;
      float best=-1;
      float x1=dataset[i*n*3+old*3+0];
      float y1=dataset[i*n*3+old*3+1];
      float z1=dataset[i*n*3+old*3+2];
      for (int k=threadIdx.x;k<n;k+=blockDim.x){
        float td=temp[blockIdx.x*n+k];
        float x2,y2,z2;
        if (k<BufferSize){
          x2=buf[k*3+0];
          y2=buf[k*3+1];
          z2=buf[k*3+2];
        }else{
          x2=dataset[i*n*3+k*3+0];
          y2=dataset[i*n*3+k*3+1];
          z2=dataset[i*n*3+k*3+2];
        }
        float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
        float d2=min(d,td);
        if (d2!=td)
          temp[blockIdx.x*n+k]=d2;
        if (d2>best){
          best=d2;
          besti=k;
        }
      }
      dists[threadIdx.x]=best;
      dists_i[threadIdx.x]=besti;
      for (int u=0;(1<<u)<blockDim.x;u++){
        __syncthreads();
        if (threadIdx.x<(blockDim.x>>(u+1))){
          int i1=(threadIdx.x*2)<<u;
          int i2=(threadIdx.x*2+1)<<u;
          if (dists[i1]<dists[i2]){
            dists[i1]=dists[i2];
            dists_i[i1]=dists_i[i2];
          }
        }
      }
      __syncthreads();
      old=dists_i[0];
      if (threadIdx.x==0)
        idxs[i*m+j]=old;
    }
  }
}

__global__ void gatherpointKernel(int b,int n,int m,const float * __restrict__ inp,const int * __restrict__ idx,float * __restrict__ out){
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
      int a=idx[i*m+j];
      out[(i*m+j)*3+0]=inp[(i*n+a)*3+0];
      out[(i*m+j)*3+1]=inp[(i*n+a)*3+1];
      out[(i*m+j)*3+2]=inp[(i*n+a)*3+2];
    }
  }
}

__global__ void scatteraddpointKernel(int b,int n,int m,const float * __restrict__ out_g,const int * __restrict__ idx,float * __restrict__ inp_g){
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
      int a=idx[i*m+j];
      atomicAdd(&inp_g[(i*n+a)*3+0],out_g[(i*m+j)*3+0]);
      atomicAdd(&inp_g[(i*n+a)*3+1],out_g[(i*m+j)*3+1]);
      atomicAdd(&inp_g[(i*n+a)*3+2],out_g[(i*m+j)*3+2]);
    }
  }
}

void cumsumLauncher(int b,int n,const float * inp,float * out){
  cumsumKernel<<<32,512>>>(b,n,inp,out);
}
//require b*n working space
void probsampleLauncher(int b,int n,int m,const float * inp_p,const float * inp_r,float * temp,int * out){
  cumsumKernel<<<32,512>>>(b,n,inp_p,temp);
  binarysearchKernel<<<dim3(32,8,1),512>>>(b,n,m,temp,inp_r,out);
}
//require 32*n working space
void farthestpointsamplingLauncher(int b,int n,int m,const float * inp,float * temp,int * out){
  farthestpointsamplingKernel<<<32,512>>>(b,n,m,inp,temp,out);
}
void gatherpointLauncher(int b,int n,int m,const float * inp,const int * idx,float * out){
  gatherpointKernel<<<dim3(2,8,1),512>>>(b,n,m,inp,idx,out);
}
void scatteraddpointLauncher(int b,int n,int m,const float * out_g,const int * idx,float * inp_g){
  scatteraddpointKernel<<<dim3(2,8,1),512>>>(b,n,m,out_g,idx,inp_g);
}


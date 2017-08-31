#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
using namespace tensorflow;

REGISTER_OP("QueryBallPoint")
    .Attr("radius: float")
    .Attr("nsample: int")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Output("idx: int32")
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int nsample;
        TF_RETURN_IF_ERROR(c->GetAttr("nsample", &nsample));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), nsample});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(1, output2);
        return Status::OK();
    });
REGISTER_OP("SelectionSort")
    .Attr("k: int")
    .Input("dist: float32")
    .Output("outi: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        return Status::OK();
    });
REGISTER_OP("GroupPoint")
    .Input("points: float32")
    .Input("idx: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * ndataset * channels
        c->WithRank(c->input(0), 3, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoints * nsample
        c->WithRank(c->input(1), 3, &dims2);
        // batch_size * npoints * nsample * channels
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), c->Dim(dims2, 2), c->Dim(dims1, 2)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("GroupPointGrad")
    .Input("points: float32")
    .Input("idx: int32")
    .Input("grad_out: float32")
    .Output("grad_points: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


void queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt);
class QueryBallPointGpuOp : public OpKernel {
    public:
        explicit QueryBallPointGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryBallPoint expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("nsample", &nsample_));
            OP_REQUIRES(context, nsample_ > 0, errors::InvalidArgument("QueryBallPoint expects positive nsample"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& xyz1_tensor = context->input(0);
            OP_REQUIRES(context, xyz1_tensor.dims()==3 && xyz1_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = xyz1_tensor.shape().dim_size(0);
            int n = xyz1_tensor.shape().dim_size(1);

            const Tensor& xyz2_tensor = context->input(1);
            OP_REQUIRES(context, xyz2_tensor.dims()==3 && xyz2_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallPoint expects (batch_size, npoint, 3) xyz2 shape."));
            int m = xyz2_tensor.shape().dim_size(1);

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,nsample_}, &idx_tensor));
            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m}, &pts_cnt_tensor));

            auto xyz1_flat = xyz1_tensor.flat<float>();
            const float *xyz1 = &(xyz1_flat(0));
            auto xyz2_flat = xyz2_tensor.flat<float>();
            const float *xyz2 = &(xyz2_flat(0));
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));
            queryBallPointLauncher(b,n,m,radius_,nsample_,xyz1,xyz2,idx,pts_cnt);
        }
    private:
        float radius_;
        int nsample_;
};
REGISTER_KERNEL_BUILDER(Name("QueryBallPoint").Device(DEVICE_GPU), QueryBallPointGpuOp);

void selectionSortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out);
class SelectionSortGpuOp : public OpKernel {
    public:
        explicit SelectionSortGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
            OP_REQUIRES(context, k_ > 0, errors::InvalidArgument("SelectionSort expects positive k"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& dist_tensor = context->input(0);
            OP_REQUIRES(context, dist_tensor.dims()==3, errors::InvalidArgument("SelectionSort expects (b,m,n) dist shape."));
            int b = dist_tensor.shape().dim_size(0);
            int m = dist_tensor.shape().dim_size(1);
            int n = dist_tensor.shape().dim_size(2);

            Tensor *outi_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,n}, &outi_tensor));
            Tensor *out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m,n}, &out_tensor));

            auto dist_flat = dist_tensor.flat<float>();
            const float *dist = &(dist_flat(0));
            auto outi_flat = outi_tensor->flat<int>();
            int *outi = &(outi_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            selectionSortLauncher(b,n,m,k_,dist,outi,out);
        }
    private:
        int k_;
};
REGISTER_KERNEL_BUILDER(Name("SelectionSort").Device(DEVICE_GPU), SelectionSortGpuOp);


void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out);
class GroupPointGpuOp: public OpKernel{
    public:
        explicit GroupPointGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("GroupPoint expects (batch_size, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupPoint expects (batch_size, npoints, nsample) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int nsample = idx_tensor.shape().dim_size(2);

            Tensor * out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,m,nsample,c}, &out_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            groupPointLauncher(b,n,c,m,nsample,points,idx,out);
        }
};
REGISTER_KERNEL_BUILDER(Name("GroupPoint").Device(DEVICE_GPU),GroupPointGpuOp);

void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points);
class GroupPointGradGpuOp: public OpKernel{
    public:
        explicit GroupPointGradGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("GroupPointGrad expects (batch_size, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupPointGrad expects (batch_size, npoints, nsample) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int nsample = idx_tensor.shape().dim_size(2);

            const Tensor& grad_out_tensor=context->input(2);
            OP_REQUIRES(context,grad_out_tensor.dims()==4 && grad_out_tensor.shape().dim_size(0)==b && grad_out_tensor.shape().dim_size(1)==m && grad_out_tensor.shape().dim_size(2)==nsample && grad_out_tensor.shape().dim_size(3)==c, errors::InvalidArgument("GroupPointGrad expects (batch_size, npoints, nsample, channel) grad_out shape"));

            Tensor * grad_points_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,c}, &grad_points_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto grad_out_flat = grad_out_tensor.flat<float>();
            const float *grad_out = &(grad_out_flat(0));
            auto grad_points_flat = grad_points_tensor->flat<float>();
            float *grad_points = &(grad_points_flat(0));
            cudaMemset(grad_points, 0, sizeof(float)*b*n*c);
            groupPointGradLauncher(b,n,c,m,nsample,grad_out,idx,grad_points);
        }
};
REGISTER_KERNEL_BUILDER(Name("GroupPointGrad").Device(DEVICE_GPU),GroupPointGradGpuOp);



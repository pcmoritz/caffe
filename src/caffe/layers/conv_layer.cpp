#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_set(this->blobs_[1]->count(), Dtype(0),
        this->blobs_[1]->mutable_cpu_diff());
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::RvForward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  this->M_ = this->num_output_ / this->group_;
  this->K_ = this->channels_ * this->kernel_h_ * this->kernel_w_ / this->group_;
  this->N_ = this->height_out_ * this->width_out_;

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* R_bottom_data = bottom[0]->cpu_inc_data();
  Dtype* R_top_data = (*top)[0]->mutable_cpu_inc_data();
  Dtype* col_data = this->col_buffer_.mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* inc_weight = this->blobs_[0]->cpu_inc_data();
  // LOG(INFO) << "M_ " << M_ << " K_ " << K_ << " N_ " << N_;
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  for (int n = 0; n < this->num_; ++n) {
    // First, im2col
    im2col_cpu(R_bottom_data + bottom[0]->offset(n), this->channels_, this->height_,
        this->width_, this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
        this->stride_h_, this->stride_w_, col_data);
    // Second, innerproduct with groups
    for (int g = 0; g < this->group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
        (Dtype)0., R_top_data + (*top)[0]->offset(n) + top_offset * g);
    }
    im2col_cpu(bottom_data + bottom[0]->offset(n), this->channels_, this->height_,
        this->width_, this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
        this->stride_h_, this->stride_w_, col_data);
    for (int g = 0; g < this->group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., inc_weight + weight_offset * g, col_data + col_offset * g,
        (Dtype)1., R_top_data + (*top)[0]->offset(n) + top_offset * g);
    }
    // third, add bias
    if (this->bias_term_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->num_output_,
          N_, 1, (Dtype)1., this->blobs_[1]->cpu_inc_data(),
          reinterpret_cast<const Dtype*>(this->bias_multiplier_.cpu_data()),
          (Dtype)1., R_top_data + (*top)[0]->offset(n));
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::RGvBackward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>& bottom) {
  const Dtype* R_top_diff = top[0]->cpu_inc_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* R_weight_diff = this->blobs_[0]->mutable_cpu_inc_diff();
  Dtype* R_bottom_diff = bottom[0]->mutable_cpu_inc_diff();
  Dtype* col_data = this->col_buffer_.mutable_cpu_data();
  Dtype* col_diff = this->col_buffer_.mutable_cpu_diff();
  if (this->bias_term_) {
    Dtype* R_bias_diff = this->blobs_[1]->mutable_cpu_inc_diff();
    memset(R_bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
    for (int n = 0; n < this->num_; ++n) {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, this->num_output_, N_,
          1., R_top_diff + top[0]->offset(n),
          reinterpret_cast<const Dtype*>(this->bias_multiplier_.cpu_data()), 1.,
          R_bias_diff);
    }
  }
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  memset(R_weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
  for (int n = 0; n < this->num_; ++n) {
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    im2col_cpu(bottom_data + bottom[0]->offset(n), this->channels_, this->height_,
        this->width_, this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
        this->stride_h_, this->stride_w_, col_data);
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int g = 0; g < this->group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., R_top_diff + top[0]->offset(n) + top_offset * g,
        col_data + col_offset * g, (Dtype)1.,
        R_weight_diff + weight_offset * g);
    }
    // gradient w.r.t. bottom data, if necessary
    if (propagate_down[0]) {
      for (int g = 0; g < this->group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + weight_offset * g,
          R_top_diff + top[0]->offset(n) + top_offset * g,
          (Dtype)0., col_diff + col_offset * g);
      }
      // col2im back to the data
      col2im_cpu(col_diff, this->channels_, this->height_, this->width_, this->kernel_h_, this->kernel_w_,
          this->pad_h_, this->pad_w_, this->stride_h_, this->stride_w_, R_bottom_diff + bottom[0]->offset(n));
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::RHvBackward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* R_top_diff = top[0]->cpu_inc_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* inc_weight = this->blobs_[0]->cpu_inc_data();
  Dtype* R_weight_diff = this->blobs_[0]->mutable_cpu_inc_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* R_bottom_data = bottom[0]->cpu_inc_data();
  Dtype* R_bottom_diff = bottom[0]->mutable_cpu_inc_diff();
  Dtype* col_data = this->col_buffer_.mutable_cpu_data();
  Dtype* col_diff = this->col_buffer_.mutable_cpu_diff();
  // bias gradient if necessary
  Dtype* R_bias_diff = NULL;
  if (this->bias_term_) {
    R_bias_diff = this->blobs_[1]->mutable_cpu_inc_diff();
    memset(R_bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
    for (int n = 0; n < this->num_; ++n) {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, this->num_output_, N_,
          1., R_top_diff + top[0]->offset(n),
          reinterpret_cast<const Dtype*>(this->bias_multiplier_.cpu_data()), 1.,
          R_bias_diff);
    }
  }
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  memset(R_weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
  for (int n = 0; n < this->num_; ++n) {
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    im2col_cpu(bottom_data + bottom[0]->offset(n), this->channels_, this->height_,
        this->width_, this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_, this->stride_h_, this->stride_w_, col_data);
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int g = 0; g < this->group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., R_top_diff + top[0]->offset(n) + top_offset * g,
        col_data + col_offset * g, (Dtype)1.,
        R_weight_diff + weight_offset * g);
    }
    im2col_cpu(R_bottom_data + bottom[0]->offset(n), this->channels_, this->height_,
        this->width_, this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_, this->stride_h_, this->stride_w_, col_data);
    for (int g = 0; g < this->group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff + top[0]->offset(n) + top_offset * g,
        col_data + col_offset * g, (Dtype)1.,
        R_weight_diff + weight_offset * g);
    }
    // gradient w.r.t. bottom data, if necessary
    if (propagate_down[0]) {
      for (int g = 0; g < this->group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + weight_offset * g,
          R_top_diff + top[0]->offset(n) + top_offset * g,
          (Dtype)0., col_diff + col_offset * g);
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., inc_weight + weight_offset * g,
          top_diff + top[0]->offset(n) + top_offset * g,
          (Dtype)1., col_diff + col_offset * g);
      }
      // col2im back to the data
      col2im_cpu(col_diff, this->channels_, this->height_, this->width_, this->kernel_h_, this->kernel_w_,
          this->pad_h_, this->pad_w_, this->stride_h_, this->stride_w_, R_bottom_diff + bottom[0]->offset(n));
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe

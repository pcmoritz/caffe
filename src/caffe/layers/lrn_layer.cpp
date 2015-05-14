#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  size_ = this->layer_param_.lrn_param().local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param_.lrn_param().alpha();
  beta_ = this->layer_param_.lrn_param().beta();
  k_ = this->layer_param_.lrn_param().k();
  scale_.Reshape(num_, channels_, height_, width_);
  if (this->layer_param_.lrn_param().norm_region() ==
    LRNParameter_NormRegion_ACROSS_CHANNELS) {
    top[0]->Reshape(num_, channels_, height_, width_);
    scale_.Reshape(num_, channels_, height_, width_); // duplicated from above
  }
  if (this->layer_param_.lrn_param().norm_region() ==
      LRNParameter_NormRegion_WITHIN_CHANNEL) {
    // Set up split_layer_ to use inputs in the numerator and denominator.
    split_top_vec_.clear();
    split_top_vec_.push_back(&product_input_);
    split_top_vec_.push_back(&square_input_);
    LayerParameter split_param;
    split_layer_.reset(new SplitLayer<Dtype>(split_param));
    split_layer_->SetUp(bottom, split_top_vec_);
    // Set up square_layer_ to square the inputs.
    square_bottom_vec_.clear();
    square_top_vec_.clear();
    square_bottom_vec_.push_back(&square_input_);
    square_top_vec_.push_back(&square_output_);
    LayerParameter square_param;
    square_param.mutable_power_param()->set_power(Dtype(2));
    square_layer_.reset(new PowerLayer<Dtype>(square_param));
    square_layer_->SetUp(square_bottom_vec_, square_top_vec_);
    // Set up pool_layer_ to sum over square neighborhoods of the input.
    pool_top_vec_.clear();
    pool_top_vec_.push_back(&pool_output_);
    LayerParameter pool_param;
    pool_param.mutable_pooling_param()->set_pool(
        PoolingParameter_PoolMethod_AVE);
    pool_param.mutable_pooling_param()->set_pad(pre_pad_);
    pool_param.mutable_pooling_param()->set_kernel_size(size_);
    pool_layer_.reset(new PoolingLayer<Dtype>(pool_param));
    pool_layer_->SetUp(square_top_vec_, pool_top_vec_);
    // Set up power_layer_ to compute (1 + alpha_/N^2 s)^-beta_, where s is
    // the sum of a squared neighborhood (the output of pool_layer_).
    power_top_vec_.clear();
    power_top_vec_.push_back(&power_output_);
    LayerParameter power_param;
    power_param.mutable_power_param()->set_power(-beta_);
    power_param.mutable_power_param()->set_scale(alpha_);
    power_param.mutable_power_param()->set_shift(Dtype(1));
    power_layer_.reset(new PowerLayer<Dtype>(power_param));
    power_layer_->SetUp(pool_top_vec_, power_top_vec_);
    // Set up a product_layer_ to compute outputs by multiplying inputs by the
    // inverse demoninator computed by the power layer.
    product_bottom_vec_.clear();
    product_bottom_vec_.push_back(&product_input_);
    product_bottom_vec_.push_back(&power_output_);
    LayerParameter product_param;
    EltwiseParameter* eltwise_param = product_param.mutable_eltwise_param();
    eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
    product_layer_.reset(new EltwiseLayer<Dtype>(product_param));
    product_layer_->SetUp(product_bottom_vec_, top);
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    top[0]->Reshape(num_, channels_, height_, width_);
    scale_.Reshape(num_, channels_, height_, width_);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    split_layer_->Reshape(bottom, split_top_vec_);
    square_layer_->Reshape(square_bottom_vec_, square_top_vec_);
    pool_layer_->Reshape(square_top_vec_, pool_top_vec_);
    power_layer_->Reshape(pool_top_vec_, power_top_vec_);
    product_layer_->Reshape(product_bottom_vec_, top);
    break;
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_cpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelForward(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelForward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  // start with the constant value
  for (int i = 0; i < scale_.count(); ++i) {
    scale_data[i] = k_;
  }
  Blob<Dtype> padded_square(1, channels_ + size_ - 1, height_, width_);
  Dtype* padded_square_data = padded_square.mutable_cpu_data();
  caffe_set(padded_square.count(), Dtype(0), padded_square_data);
  Dtype alpha_over_size = alpha_ / size_;
  // go through the images
  for (int n = 0; n < num_; ++n) {
    // compute the padded square
    caffe_sqr(channels_ * height_ * width_,
        bottom_data + bottom[0]->offset(n),
        padded_square_data + padded_square.offset(0, pre_pad_));
    // Create the first channel scale
    for (int c = 0; c < size_; ++c) {
      caffe_axpy<Dtype>(height_ * width_, alpha_over_size,
          padded_square_data + padded_square.offset(0, c),
          scale_data + scale_.offset(n, 0));
    }
    for (int c = 1; c < channels_; ++c) {
      // copy previous scale
      caffe_copy<Dtype>(height_ * width_,
          scale_data + scale_.offset(n, c - 1),
          scale_data + scale_.offset(n, c));
      // add head
      caffe_axpy<Dtype>(height_ * width_, alpha_over_size,
          padded_square_data + padded_square.offset(0, c + size_ - 1),
          scale_data + scale_.offset(n, c));
      // subtract tail
      caffe_axpy<Dtype>(height_ * width_, -alpha_over_size,
          padded_square_data + padded_square.offset(0, c - 1),
          scale_data + scale_.offset(n, c));
    }
  }

  // In the end, compute output
  caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, top_data);
  caffe_mul<Dtype>(scale_.count(), top_data, bottom_data, top_data);
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelForward(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  split_layer_->Forward(bottom, split_top_vec_);
  square_layer_->Forward(square_bottom_vec_, square_top_vec_);
  pool_layer_->Forward(square_top_vec_, pool_top_vec_);
  power_layer_->Forward(pool_top_vec_, power_top_vec_);
  product_layer_->Forward(product_bottom_vec_, top);
}

template <typename Dtype>
void LRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // const Dtype* top_data = top[0]->cpu_data();
    // const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* scale_data = scale_.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    switch (this->layer_param_.lrn_param().norm_region()) {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
      // CrossChannelBackward_cpu(top, top_diff, top_data, bottom_data, scale_data,
      //  bottom_diff);
      break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
      // WithinChannelBackward(top, propagate_down, bottom);
      break;
    default:
      LOG(FATAL) << "Unknown normalization region.";
    }
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_cpu(const vector<Blob<Dtype>*>& top,
    const Dtype* top_diff, const Dtype* top_data, const Dtype* bottom_data,
    const Dtype* scale_data, Dtype* bottom_diff) {
  Blob<Dtype> padded_ratio(1, channels_ + size_ - 1, height_, width_);
  Blob<Dtype> accum_ratio(1, 1, height_, width_);
  Dtype* padded_ratio_data = padded_ratio.mutable_cpu_data();
  Dtype* accum_ratio_data = accum_ratio.mutable_cpu_data();
  // We hack a little bit by using the diff() to store an additional result
  Dtype* accum_ratio_times_bottom = accum_ratio.mutable_cpu_diff();
  caffe_set(padded_ratio.count(), Dtype(0), padded_ratio_data);
  Dtype cache_ratio_value = 2. * alpha_ * beta_ / size_;

  caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, bottom_diff);
  caffe_mul<Dtype>(scale_.count(), top_diff, bottom_diff, bottom_diff);

  // go through individual data
  int inverse_pre_pad = size_ - (size_ + 1) / 2;
  for (int n = 0; n < num_; ++n) {
    int block_offset = scale_.offset(n);
    // first, compute diff_i * y_i / s_i
    caffe_mul<Dtype>(channels_ * height_ * width_,
        top_diff + block_offset, top_data + block_offset,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
    caffe_div<Dtype>(channels_ * height_ * width_,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad),
        scale_data + block_offset,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
    // Now, compute the accumulated ratios and the bottom diff
    caffe_set(accum_ratio.count(), Dtype(0), accum_ratio_data);
    for (int c = 0; c < size_ - 1; ++c) {
      caffe_axpy<Dtype>(height_ * width_, 1.,
          padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
    }
    for (int c = 0; c < channels_; ++c) {
      caffe_axpy<Dtype>(height_ * width_, 1.,
          padded_ratio_data + padded_ratio.offset(0, c + size_ - 1),
          accum_ratio_data);
      // compute bottom diff
      caffe_mul<Dtype>(height_ * width_,
          bottom_data + top[0]->offset(n, c),
          accum_ratio_data, accum_ratio_times_bottom);
      caffe_axpy<Dtype>(height_ * width_, -cache_ratio_value,
          accum_ratio_times_bottom, bottom_diff + top[0]->offset(n, c));
      caffe_axpy<Dtype>(height_ * width_, -1.,
          padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
    }
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelBackward(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    vector<bool> product_propagate_down(2, true);
    product_layer_->Backward(top, product_propagate_down, product_bottom_vec_);
    power_layer_->Backward(power_top_vec_, propagate_down, pool_top_vec_);
    pool_layer_->Backward(pool_top_vec_, propagate_down, square_top_vec_);
    square_layer_->Backward(square_top_vec_, propagate_down,
                            square_bottom_vec_);
    split_layer_->Backward(split_top_vec_, propagate_down, bottom);
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::RvForward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelRvForward_cpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelRvForward(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelRvForward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const Dtype* top_diff = (*top)[0]->cpu_diff();
  const Dtype* top_data = (*top)[0]->cpu_data();
  const Dtype* R_bottom_data = bottom[0]->cpu_inc_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scale_data = scale_.cpu_data();
  const Dtype* bottom_diff = bottom[0]->cpu_diff();
  // Hack: use scale_diff to store summation \sum_j[x_j * R{x_j}]
  Dtype* scale_diff = scale_.mutable_cpu_diff();
  Dtype* R_top_data = (*top)[0]->mutable_cpu_inc_data();
  // Hack: use R_top_diff to store second term
  Dtype* R_top_data_second_term = (*top)[0]->mutable_cpu_inc_diff();
  Blob<Dtype> padded_ratio(1, channels_ + size_ - 1, height_, width_);
  Dtype* padded_ratio_data = padded_ratio.mutable_cpu_data();
  memset(padded_ratio_data, 0, sizeof(Dtype) * padded_ratio.count());
  Dtype cache_ratio_value = 2. * alpha_ * beta_ / size_;
  Blob<Dtype> padded_square(1, channels_ + size_ - 1, height_, width_);
  Dtype* padded_square_data = padded_square.mutable_cpu_data();
  memset(padded_square_data, 0, sizeof(Dtype) * padded_square.count());
  memset(scale_diff, 0, sizeof(Dtype) * scale_.count());

  // Compute s_i^{-\beta}
  caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, R_top_data);
  // Compute R{x_i} * s_i^{-\beta}
  caffe_mul<Dtype>(scale_.count(), R_bottom_data, R_top_data, R_top_data);

  // Go through images to compute scale_diff[i] <- \sum_j[x_j R{x_j}]
  int inverse_pre_pad = size_ - (size_ + 1) / 2;
  for (int n = 0; n < num_; ++n) {
    int block_offset = bottom[0]->offset(n);
    caffe_mul(channels_ * height_ * width_,
        bottom_data + block_offset,
        R_bottom_data + block_offset,
        padded_square_data + padded_square.offset(0, pre_pad_));
    // Create the first channel scale
    for (int c = 0; c < size_; ++c) {
      caffe_axpy<Dtype>(height_ * width_, Dtype(1),
          padded_square_data + padded_square.offset(0, c),
          scale_diff + scale_.offset(n, 0));
    }
    for (int c = 1; c < channels_; ++c) {
      // copy previous scale
      caffe_copy<Dtype>(height_ * width_,
          scale_diff + scale_.offset(n, c - 1),
          scale_diff + scale_.offset(n, c));
      // add head
      caffe_axpy<Dtype>(height_ * width_, Dtype(1),
          padded_square_data + padded_square.offset(0, c + size_ - 1),
          scale_diff + scale_.offset(n, c));
      // subtract tail
      caffe_axpy<Dtype>(height_ * width_, Dtype(-1),
          padded_square_data + padded_square.offset(0, c - 1),
          scale_diff + scale_.offset(n, c));
    }
  }

  // Compute \sum_j[x_j * R{x_j}] * y_i
  caffe_mul<Dtype>(scale_.count(), scale_diff, top_data,
      R_top_data_second_term);
  // Compute \sum_j[x_j * R{x_j}] * y_i / s_i
  caffe_div<Dtype>(scale_.count(), R_top_data_second_term, scale_data,
      R_top_data_second_term);
  // Compute R{y_i}
  caffe_axpy<Dtype>(scale_.count(), -cache_ratio_value, R_top_data_second_term,
      R_top_data);
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelRvForward(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LOG(INFO) << "before split layer";
  split_layer_->RvForward(bottom, &split_top_vec_);
  LOG(INFO) << "before square layer";
  square_layer_->RvForward(square_bottom_vec_, &square_top_vec_);
  LOG(INFO) << "before pool layer";
  pool_layer_->RvForward(square_top_vec_, &pool_top_vec_);
  LOG(INFO) << "before power layer";
  power_layer_->RvForward(pool_top_vec_, &power_top_vec_);
  LOG(INFO) << "before product layer";
  product_layer_->RvForward(product_bottom_vec_, top);
}

template <typename Dtype>
void LRNLayer<Dtype>::RGvBackward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS: {
      const Dtype* R_top_diff = top[0]->cpu_inc_diff();
      const Dtype* top_data = top[0]->cpu_data();
      const Dtype* bottom_data = bottom[0]->cpu_data();
      const Dtype* scale_data = scale_.cpu_data();
      Dtype* R_bottom_diff = bottom[0]->mutable_cpu_inc_diff();
      CrossChannelBackward_cpu(top, R_top_diff, top_data, bottom_data,
          scale_data, R_bottom_diff);
    }
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelRGvBackward(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelRGvBackward(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    product_layer_->RGvBackward(top, propagate_down, product_bottom_vec_);
    power_layer_->RGvBackward(power_top_vec_, propagate_down, pool_top_vec_);
    pool_layer_->RGvBackward(pool_top_vec_, propagate_down, square_top_vec_);
    square_layer_->RGvBackward(square_top_vec_, propagate_down,
                               square_bottom_vec_);
    split_layer_->RGvBackward(split_top_vec_, propagate_down, bottom);
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelRHvBackward(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    product_layer_->RHvBackward(top, propagate_down, product_bottom_vec_);
    power_layer_->RHvBackward(power_top_vec_, propagate_down, pool_top_vec_);
    pool_layer_->RHvBackward(pool_top_vec_, propagate_down, square_top_vec_);
    square_layer_->RHvBackward(square_top_vec_, propagate_down,
                               square_bottom_vec_);
    split_layer_->RHvBackward(split_top_vec_, propagate_down, bottom);
  }
}

#ifdef CPU_ONLY
STUB_GPU(LRNLayer);
STUB_GPU_FORWARD(LRNLayer, CrossChannelForward);
STUB_GPU_BACKWARD(LRNLayer, CrossChannelBackward);
#endif

INSTANTIATE_CLASS(LRNLayer);
REGISTER_LAYER_CLASS(LRN);

}  // namespace caffe

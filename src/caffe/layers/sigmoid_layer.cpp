#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::RvForward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* top_data = (*top)[0]->cpu_data();
  const Dtype* R_bottom_data = bottom[0]->cpu_inc_data();
  Dtype* R_top_data = (*top)[0]->mutable_cpu_inc_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    R_top_data[i] = R_bottom_data[i] * top_data[i] * (Dtype(1) - top_data[i]);
  }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::RGvBackward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* R_top_diff = top[0]->cpu_inc_diff();
    Dtype* R_bottom_diff = bottom[0]->mutable_cpu_inc_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      Dtype sigmoid_x = top_data[i];
      R_bottom_diff[i] = R_top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::RHvBackward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* R_top_data = top[0]->cpu_inc_data();
    const Dtype* R_top_diff = top[0]->cpu_inc_diff();
    Dtype* R_bottom_diff = bottom[0]->mutable_cpu_inc_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      R_bottom_diff[i] = R_top_diff[i]
          * top_data[i] * (1 - top_data[i])
          + top_diff[i] * R_top_data[i] * (1 - 2 * top_data[i]);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLayer);
#endif

INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe

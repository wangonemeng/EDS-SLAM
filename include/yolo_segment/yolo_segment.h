#pragma once

#include <string>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvInferPlugin.h"

struct SegOutput {
  int class_id;
  std::string class_name;
  float confidence;
  cv::Rect box;
  cv::Mat boxMask;
};

class YoloSeg {
 public:
  YoloSeg(std::string model_path, std::string model_name);
  ~YoloSeg();
  std::vector<SegOutput> SegmentImage(const cv::Mat& image) const;
  std::vector<SegOutput> SegmentDynamic(const cv::Mat& image) const;
  cv::Mat DrawSegement(const cv::Mat& img, const std::vector<SegOutput>& seg_result) const;

 private:
  void Inference(float* input, float* output, float* output1) const;
  bool onnx2trt() const;

 public:
  const std::vector<std::string> class_names = {
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
      "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
      "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
      "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
      "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
      "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
      "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
      "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
      "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
      "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
      "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
      "teddy bear", "hair drier", "toothbrush"};
  const std::vector<std::string> dynamic_class_names = {
      "person", "bicycle", "car", "motorcycle", "bus", "truck"};

 private:
  const std::string model_path_;
  const std::string model_name_;

  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;

  static constexpr int input_h_ = 640;
  static constexpr int input_w_ = 640;
  static constexpr int seg_w_ = 160;
  static constexpr int seg_h_ = 160;
  static constexpr int classes_ = 80;
  static constexpr int batch_size_ = 1;
  static constexpr int num_box_ = 25200;
  static constexpr int seg_channels_ = 32;
  static constexpr int output_size_ = num_box_ * (classes_ + 5 + seg_channels_);
  static constexpr int output_size1_ = seg_channels_ * seg_w_ * seg_h_;
  static constexpr float CONF_THRESHOLD = 0.1;
  static constexpr float NMS_THRESHOLD = 0.5;
  static constexpr float MASK_THRESHOLD = 0.5;
};

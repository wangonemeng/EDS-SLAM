#include "yolo_segment.h"

#include "logging.h"
#include "utils.h"

#include <chrono>
#include <unistd.h>
#include "NvOnnxParser.h"

using namespace nvinfer1;

YoloSeg::YoloSeg(std::string model_path, std::string model_name)
    : model_path_(model_path), model_name_(model_name) {
  std::string trt_path = model_path + "/" + model_name + ".trt";
  if (access(trt_path.c_str(), F_OK) == -1) {
    onnx2trt();
  }
  assert(access(trt_path.c_str(), F_OK) != -1);
  char* trtModelStream{nullptr};
  size_t size{0};
  std::ifstream file(trt_path.c_str(), std::ios::binary);
  if (file.good()) {
    std::cout << "load trt success: " << trt_path << std::endl;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
  } else {
    std::cout << "load trt failed: " << trt_path << std::endl;
  }

  Logger gLogger;
  runtime_ = createInferRuntime(gLogger);
  assert(runtime_ != nullptr);
  bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
  engine_ = runtime_->deserializeCudaEngine(trtModelStream, size, nullptr);
  assert(engine_ != nullptr);
  context_ = engine_->createExecutionContext();
  assert(context_ != nullptr);
  delete[] trtModelStream;
}

YoloSeg::~YoloSeg() {
  if (context_) context_->destroy();
  if (engine_) engine_->destroy();
  if (runtime_) runtime_->destroy();
}

std::vector<SegOutput> YoloSeg::SegmentImage(const cv::Mat& src) const {
  if (src.empty()) return std::vector<SegOutput>();
  cv::Mat image = src;
  if (image.channels() == 1) {
    cv::cvtColor(src, image, cv::COLOR_GRAY2BGR);
  }
  static float data[3 * input_h_ * input_w_];
  cv::Mat pr_img;
  std::vector<int> padsize;
  pr_img = preprocess_img(image, input_h_, input_w_, padsize);
  int i = 0;
  for (int row = 0; row < input_h_; ++row) {
    uchar* uc_pixel = pr_img.data + row * pr_img.step;
    for (int col = 0; col < input_w_; ++col) {
      data[i] = (float)uc_pixel[2] / 255.0;
      data[i + input_h_ * input_w_] = (float)uc_pixel[1] / 255.0;
      data[i + 2 * input_h_ * input_w_] = (float)uc_pixel[0] / 255.0;
      uc_pixel += 3;
      ++i;
    }
  }
  static float prob[output_size_];
  static float prob1[output_size1_];
  const auto& t1 = std::chrono::steady_clock::now();
  Inference(data, prob, prob1);
  const auto& t2 = std::chrono::steady_clock::now();
  double dt1 = std::chrono::duration<double>(t2 - t1).count() * 1000.;
  std::cout << "Yolo Inference cost: " << dt1 << " ms" << std::endl;

  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<float>> picked_proposals;

  int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
  float ratio_h = (float)image.rows / newh;
  float ratio_w = (float)image.cols / neww;

  int net_width = classes_ + 5 + seg_channels_;
  float* pdata = prob;
  for (int j = 0; j < num_box_; ++j) {
    float box_score = pdata[4];
    if (box_score >= CONF_THRESHOLD) {
      cv::Mat scores(1, classes_, CV_32FC1, pdata + 5);
      cv::Point classIdPoint;
      double max_class_socre;
      minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
      max_class_socre = (float)max_class_socre;
      if (max_class_socre >= CONF_THRESHOLD) {
        std::vector<float> temp_proto(pdata + 5 + classes_, pdata + net_width);
        picked_proposals.push_back(temp_proto);

        float x = (pdata[0] - padw) * ratio_w;  // x
        float y = (pdata[1] - padh) * ratio_h;  // y
        float w = pdata[2] * ratio_w;           // w
        float h = pdata[3] * ratio_h;           // h

        int left = MAX((x - 0.5 * w), 0);
        int top = MAX((y - 0.5 * h), 0);
        classIds.push_back(classIdPoint.x);
        confidences.push_back(max_class_socre * box_score);
        boxes.push_back(cv::Rect(left, top, int(w), int(h)));
      }
    }
    pdata += net_width;
  }
  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, nms_result);
  if (nms_result.empty()) return std::vector<SegOutput>();
  std::vector<std::vector<float>> temp_mask_proposals;
  cv::Rect holeImgRect(0, 0, image.cols, image.rows);
  std::vector<SegOutput> output;
  for (int i = 0; i < nms_result.size(); ++i) {
    int idx = nms_result[i];
    SegOutput result;
    result.class_id = classIds[idx];
    result.class_name = class_names[result.class_id];
    result.confidence = confidences[idx];
    result.box = boxes[idx] & holeImgRect;
    output.push_back(result);
    temp_mask_proposals.push_back(picked_proposals[idx]);
  }

  cv::Mat maskProposals;
  for (int i = 0; i < temp_mask_proposals.size(); ++i)
    maskProposals.push_back(cv::Mat(temp_mask_proposals[i]).t());

  pdata = prob1;
  std::vector<float> mask(pdata, pdata + seg_channels_ * seg_w_ * seg_h_);

  cv::Mat mask_protos = cv::Mat(mask);
  cv::Mat protos = mask_protos.reshape(0, {seg_channels_, seg_w_ * seg_h_});

  cv::Mat matmulRes = (maskProposals * protos).t();
  cv::Mat masks = matmulRes.reshape(output.size(), {seg_w_, seg_h_});
  std::vector<cv::Mat> maskChannels;
  split(masks, maskChannels);
  for (int i = 0; i < output.size(); ++i) {
    cv::Mat dest, mask;
    cv::exp(-maskChannels[i], dest);
    dest = 1.0 / (1.0 + dest);  // 160*160
    cv::Rect roi(int((float)padw / input_w_ * seg_w_), int((float)padh / input_h_ * seg_h_),
                 int(seg_w_ - padw / 2), int(seg_h_ - padh / 2));
    dest = dest(roi);
    cv::resize(dest, mask, image.size(), cv::INTER_NEAREST);
    cv::Rect temp_rect = output[i].box;
    mask = mask(temp_rect) > MASK_THRESHOLD;
    output[i].boxMask = mask;
  }

  return output;
}

std::vector<SegOutput> YoloSeg::SegmentDynamic(const cv::Mat& image) const {
  const std::vector<SegOutput>& seg_outputs = SegmentImage(image);
  std::vector<SegOutput> dynamic_outputs;
  for (const auto& seg : seg_outputs) {
    if (find(dynamic_class_names.begin(), dynamic_class_names.end(), seg.class_name) !=
        dynamic_class_names.end()) {
      dynamic_outputs.emplace_back(seg);
    }
  }
  return dynamic_outputs;
}

cv::Mat YoloSeg::DrawSegement(const cv::Mat& img, const std::vector<SegOutput>& seg_result) const {
  std::vector<cv::Scalar> color;
  srand(time(0));
  for (int i = 0; i < classes_; i++) {
    int b = rand() % 256;
    int g = rand() % 256;
    int r = rand() % 256;
    color.push_back(cv::Scalar(b, g, r));
  }
  cv::Mat mask = img.clone();
  cv::Mat rgb = img.clone();
  if (img.channels() == 1) {
    cv::cvtColor(img, rgb, cv::COLOR_GRAY2BGR);
  }
  for (int i = 0; i < seg_result.size(); i++) {
    int left, top;
    left = seg_result[i].box.x;
    top = seg_result[i].box.y;
    int color_num = i;
    //cv::rectangle(rgb, seg_result[i].box, color[seg_result[i].class_id], 2, 8);
    mask(seg_result[i].box).setTo(color[seg_result[i].class_id], seg_result[i].boxMask);

    std::string label = seg_result[i].class_name + ":" + std::to_string(seg_result[i].confidence);
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    //cv::putText(rgb, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color[seg_result[i].class_id], 2);
  }
  cv::addWeighted(rgb, 0.5, mask, 0.5, 0, rgb);
  return rgb;
}

void YoloSeg::Inference(float* input, float* output, float* output1) const {
  const ICudaEngine& engine = context_->getEngine();

  const char* INPUT_BLOB_NAME = "images";
  const char* OUTPUT_BLOB_NAME = "output0";   // detect
  const char* OUTPUT_BLOB_NAME1 = "output1";  // mask

  // Pointers to input and output device buffers to pass to engine.
  // Engine requires exactly IEngine::getNbBindings() number of buffers.
  assert(engine.getNbBindings() == 3);
  void* buffers[3];

  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
  const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
  const int outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1);
  // Create GPU buffers on device
  CHECK(cudaMalloc(&buffers[inputIndex], batch_size_ * 3 * input_h_ * input_w_ * sizeof(float)));  //
  CHECK(cudaMalloc(&buffers[outputIndex], batch_size_ * output_size_ * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex1], batch_size_ * output_size1_ * sizeof(float)));

  // Create stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batch_size_ * 3 * input_h_ * input_w_ * sizeof(float), cudaMemcpyHostToDevice, stream));
  context_->enqueue(batch_size_, buffers, stream, nullptr);
  CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batch_size_ * output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CHECK(cudaMemcpyAsync(output1, buffers[outputIndex1], batch_size_ * output_size1_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
  CHECK(cudaFree(buffers[outputIndex1]));
}

bool YoloSeg::onnx2trt() const {
  std::string onnx_path = model_path_ + "/" + model_name_ + ".onnx";
  std::string trt_path = model_path_ + "/" + model_name_ + ".trt";
  if (access(onnx_path.c_str(), F_OK) != -1) {
    std::cout << "onnx exist: " << onnx_path << std::endl;
  } else {
    std::cout << "onnx not exist: " << onnx_path << std::endl;
    return false;
  }

  Logger gLogger;
  IBuilder* builder = createInferBuilder(gLogger);
  const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

  parser->parseFromFile(onnx_path.c_str(), static_cast<int>(Logger::Severity::kWARNING));
  for (int i = 0; i < parser->getNbErrors(); ++i) {
    std::cout << parser->getError(i)->desc() << std::endl;
  }
  std::cout << "successfully load the onnx model" << std::endl;

  std::cout << "Convert ONNX to TRT ......" << std::endl;
  unsigned int maxBatchSize = 1;
  builder->setMaxBatchSize(maxBatchSize);
  IBuilderConfig* config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(128 * (1 << 20));
  config->setFlag(BuilderFlag::kFP16);
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

  IHostMemory* gieModelStream = engine->serialize();
  std::ofstream p(trt_path.c_str(), std::ios::binary);
  if (!p) {
    std::cerr << "could not open trt file: " << trt_path << std::endl;
    return -1;
  }
  p.write(reinterpret_cast<const char*>(gieModelStream->data()), gieModelStream->size());
  gieModelStream->destroy();

  std::cout << "Convert ONNX to TRT Success" << std::endl;

  return true;
}

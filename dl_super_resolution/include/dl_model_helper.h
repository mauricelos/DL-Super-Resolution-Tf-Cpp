//  dl_model_helper.h

#ifndef DL_SUPER_RESOLUTION_DL_MODEL_HELPER_H
#define DL_SUPER_RESOLUTION_DL_MODEL_HELPER_H

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

class DlModelHelper
{
  public:
    DlModelHelper() = default;

    tensorflow::Status CreateTensorFromImage(const std::string& image_file_name,
                                             std::vector<tensorflow::Tensor>& tensor_container,
                                             const std::array<std::uint32_t, 3>& tensor_dimsensions = {1080U,
                                                                                                       1920U,
                                                                                                       3U});

  private:
    inline bool EndsWith(std::string const& file_name, std::string const& file_suffix)
    {
        return (file_suffix.size() < file_name.size())
                   ? std::equal(file_suffix.rbegin(), file_suffix.rend(), file_name.rbegin())
                   : false;
    }

    tensorflow::Status CreateBatchFromTensors(std::uint32_t& batch_size,
                                              std::vector<tensorflow::Tensor>& input_tensor_container_ground_truth,
                                              std::vector<tensorflow::Tensor>& input_tensor_container_down_sampled,
                                              std::vector<tensorflow::Tensor>& batch_tensor_container);

    tensorflow::Status ReadEntireFile(tensorflow::Env* env, const std::string& filename, tensorflow::Tensor* output);

    const std::string image_file_reader_{"image_file_reader"};
    const std::string image_png_reader_{"image_png_reader"};
    const std::string image_jpeg_reader_{"image_jpeg_reader"};
    const std::string float_caster_{"float_caster"};
    const std::string dim_expander_{"dimension_expander"};
    const std::string image_resizer_{"bilinear_image_resizer"};
    const std::string stack_ground_truth_container_{"stack_ground_truth_container"};
    const std::string stack_down_sampled_container_{"stack_down_sampled_container"};
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
};

#endif  // DL_SUPER_RESOLUTION_DL_MODEL_HELPER_H
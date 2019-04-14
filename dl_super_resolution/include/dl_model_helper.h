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
    DlModelHelper(std::uint32_t input_height = 1080U,
                  std::uint32_t input_width = 1920U,
                  std::uint32_t num_required_image_channels = 3U)
        : input_height_(input_height),
          input_width_(input_width),
          num_required_image_channels_(num_required_image_channels)
    {
    }

    tensorflow::Status CreateTensorFromImage(const std::string& image_file_name,
                                             std::vector<tensorflow::Tensor>& tensor_container);

  private:
    inline bool EndsWith(std::string const& file_name, std::string const& file_suffix)
    {
        return (file_suffix.size() < file_name.size())
                   ? std::equal(file_suffix.rbegin(), file_suffix.rend(), file_name.rbegin())
                   : false;
    }

    tensorflow::Status ReadEntireFile(tensorflow::Env* env, const std::string& filename, tensorflow::Tensor* output);

    const std::string image_file_reader_{"image_file_reader"};
    const std::string image_png_reader_{"image_png_reader"};
    const std::string image_jpeg_reader_{"image_jpeg_reader"};
    const std::string float_caster_{"float_caster"};
    const std::string dim_expander_{"dimension_expander"};
    const std::string image_resizer_{"bilinear_image_resizer"};
    std::uint32_t input_height_;
    std::uint32_t input_width_;
    std::uint32_t num_required_image_channels_;
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
};

#endif  // DL_SUPER_RESOLUTION_DL_MODEL_HELPER_H
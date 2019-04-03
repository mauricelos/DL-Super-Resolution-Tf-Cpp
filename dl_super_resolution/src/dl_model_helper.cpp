//  dl_model_helper.cpp

#include "dl_super_resolution/include/dl_model_helper.h"

tensorflow::Status DlModelHelper::CreateTensorFromImage(const std::string& image_file_name,
                                                        std::vector<tensorflow::Tensor>* tensor_container)
{
    tensorflow::Output image_reader;
    auto image_processor_scope = tensorflow::Scope::NewRootScope();
    auto image_file_reader =
        tensorflow::ops::ReadFile(image_processor_scope.WithOpName(image_file_reader_), image_file_name);

    if (EndsWith(image_file_name, ".png"))
    {
        image_reader = tensorflow::ops::DecodePng(image_processor_scope.WithOpName(image_png_reader_),
                                                  image_file_reader,
                                                  tensorflow::ops::DecodePng::Channels(num_required_image_channels_));
    }
    else if (EndsWith(image_file_name, ".jpeg"))
    {
        image_reader = tensorflow::ops::DecodeJpeg(image_processor_scope.WithOpName(image_jpeg_reader_),
                                                   image_file_reader,
                                                   tensorflow::ops::DecodeJpeg::Channels(num_required_image_channels_));
    }
    else
    {
        LOG(INFO) << "Wrong image type. Please use .png or .jpeg!";
    }

    auto float_caster =
        tensorflow::ops::Cast(image_processor_scope.WithOpName(float_caster_), image_file_reader, tensorflow::DT_FLOAT);

    auto dimension_expanded_tensor = tensorflow::ops::ExpandDims(image_processor_scope, float_caster, 0);

    auto resized_tensor = tensorflow::ops::ResizeBilinear(
        image_processor_scope,
        dimension_expanded_tensor,
        tensorflow::ops::GuaranteeConst(image_processor_scope.WithOpName(bilinear_image_resizer_),
                                        {input_height_, input_width_}));

    tensorflow::ops::Unstack(image_processor_scope.WithOpName(finished_tensor_), resized_tensor, 1);

    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(image_processor_scope.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({}, {finished_tensor_}, {}, tensor_container));

    return tensorflow::Status::OK();
}
//  dl_model_helper.cpp

#include "dl_super_resolution/include/dl_model_helper.h"

tensorflow::Status DlModelHelper::CreateTensorFromImage(const std::string& image_file_name,
                                                        std::vector<tensorflow::Tensor>* tensor_container)
{
    auto image_file_reader =
        tensorflow::ops::ReadFile(image_processor_scope_.WithOpName(image_file_reader_), image_file_name);

    if (EndsWith(image_file_name, ".png"))
    {
        image_reader_ = tensorflow::ops::DecodePng(image_processor_scope_.WithOpName(image_png_reader_),
                                                   image_file_reader,
                                                   tensorflow::ops::DecodePng::Channels(num_required_image_channels_));
    }
    else if (EndsWith(image_file_name, ".jpeg"))
    {
        image_reader_ =
            tensorflow::ops::DecodeJpeg(image_processor_scope_.WithOpName(image_jpeg_reader_),
                                        image_file_reader,
                                        tensorflow::ops::DecodeJpeg::Channels(num_required_image_channels_));
    }
    else
    {
        LOG(INFO) << "Wrong image type. Please use .png or .jpeg!";
    }

    auto float_caster = tensorflow::ops::Cast(
        image_processor_scope_.WithOpName(float_caster_), image_file_reader, tensorflow::DT_FLOAT);

    auto dimension_expanded_tensor = tensorflow::ops::ExpandDims(image_processor_scope_, float_caster, 0);

    auto resized_tensor = tensorflow::ops::ResizeBilinear(
        image_processor_scope_,
        dimension_expanded_tensor,
        tensorflow::ops::GuaranteeConst(image_processor_scope_.WithOpName(bilinear_image_resizer_),
                                        {input_height_, input_width_}));

    auto normalized_tensor = Normalizer(image_processor_scope_, resized_tensor, tensorflow::DataType::DT_FLOAT);

    tensorflow::ops::Unstack(image_processor_scope_.WithOpName(finished_tensor_), normalized_tensor, 1);

    TF_RETURN_IF_ERROR(image_processor_scope_.ToGraphDef(&graph_));

    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph_));
    TF_RETURN_IF_ERROR(session->Run({}, {finished_tensor_}, {}, tensor_container));

    return tensorflow::Status::OK();
}

tensorflow::Output DlModelHelper::Normalizer(const tensorflow::Scope& scope,
                                             tensorflow::Input input_tensor,
                                             tensorflow::DataType dtype)
{
    tensorflow::Output output{};
    tensorflow::Node* ret;
    auto node_out_input_tensor = tensorflow::ops::AsNodeOut(scope, input_tensor);
    const auto unique_name = scope.GetUniqueNameForOp(tensor_normalizer_);
    auto builder =
        tensorflow::NodeBuilder(unique_name, tensor_normalizer_).Input(node_out_input_tensor).Attr("dtype", dtype);
    scope.UpdateBuilder(&builder);
    scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
    scope.UpdateStatus(scope.DoShapeInference(ret));
    output = tensorflow::Output(ret, 0);

    return output;
}
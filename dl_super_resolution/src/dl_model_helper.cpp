//  dl_model_helper.cpp

#include <dl_model_helper.h>

tensorflow::Status DlModelHelper::CreateTensorFromImage(const std::string& image_file_name,
                                                        std::vector<tensorflow::Tensor>& tensor_container)
{
    tensorflow::GraphDef graph;
    tensorflow::Output image_reader;
    const auto image_processor_scope = tensorflow::Scope::NewRootScope();

    tensorflow::Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(ReadEntireFile(tensorflow::Env::Default(), image_file_name, &input));
    inputs = {{image_file_reader_, input}};

    auto image_file_reader = tensorflow::ops::Placeholder(image_processor_scope.WithOpName(image_file_reader_),
                                                          tensorflow::DataType::DT_STRING);

    if (EndsWith(image_file_name, ".png"))
    {
        image_reader = tensorflow::ops::DecodePng(image_processor_scope.WithOpName(image_png_reader_),
                                                  std::move(image_file_reader),
                                                  tensorflow::ops::DecodePng::Channels(num_required_image_channels_));
    }
    else if (EndsWith(image_file_name, ".jpeg"))
    {
        image_reader = tensorflow::ops::DecodeJpeg(image_processor_scope.WithOpName(image_jpeg_reader_),
                                                   std::move(image_file_reader),
                                                   tensorflow::ops::DecodeJpeg::Channels(num_required_image_channels_));
    }
    else
    {
        LOG(INFO) << "Wrong image type. Please use .png or .jpeg!";
    }

    auto float_caster = tensorflow::ops::Cast(
        image_processor_scope.WithOpName(float_caster_), std::move(image_reader), tensorflow::DT_FLOAT);

    auto dimension_expanded_tensor =
        tensorflow::ops::ExpandDims(image_processor_scope.WithOpName(dim_expander_), std::move(float_caster), 0);

    auto resized_tensor = tensorflow::ops::ResizeBilinear(
        image_processor_scope.WithOpName(image_resizer_),
        std::move(dimension_expanded_tensor),
        tensorflow::ops::Const(image_processor_scope,
                               {static_cast<std::int32_t>(input_height_), static_cast<std::int32_t>(input_width_)}));

    TF_RETURN_IF_ERROR(image_processor_scope.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {image_resizer_}, {}, &tensor_container));

    return tensorflow::Status::OK();
}

tensorflow::Status DlModelHelper::ReadEntireFile(tensorflow::Env* env,
                                                 const std::string& filename,
                                                 tensorflow::Tensor* output)
{
    tensorflow::uint64 file_size{0};
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    std::string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));

    if (data.size() != file_size)
    {
        return tensorflow::errors::DataLoss(
            "Truncated read of '", filename, "' expected ", file_size, " got ", data.size());
    }

    output->scalar<std::string>()() = std::string(data);

    return tensorflow::Status::OK();
}
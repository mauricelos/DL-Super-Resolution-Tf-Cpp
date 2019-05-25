//  dl_model_helper.cpp

#include <dl_model_helper.h>

tensorflow::Status DlModelHelper::CreateTensorFromImage(const std::string& image_file_name,
                                                        std::vector<tensorflow::Tensor>& tensor_container,
                                                        const std::array<std::uint32_t, 3>& tensor_dimsensions)
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
                                                  tensorflow::ops::DecodePng::Channels(tensor_dimsensions[2]));
    }
    else if (EndsWith(image_file_name, ".jpeg"))
    {
        image_reader = tensorflow::ops::DecodeJpeg(image_processor_scope.WithOpName(image_jpeg_reader_),
                                                   std::move(image_file_reader),
                                                   tensorflow::ops::DecodeJpeg::Channels(tensor_dimsensions[2]));
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
        tensorflow::ops::Const(
            image_processor_scope,
            {static_cast<std::int32_t>(tensor_dimsensions[0]), static_cast<std::int32_t>(tensor_dimsensions[1])}));

    tensorflow::ops::Div(image_processor_scope.WithOpName(image_normalizer_),
                         tensorflow::ops::Sub(image_processor_scope, std::move(resized_tensor), {tensor_mean_}),
                         {tensor_std_});

    TF_RETURN_IF_ERROR(image_processor_scope.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {image_normalizer_}, {}, &tensor_container));

    return tensorflow::Status::OK();
}

tensorflow::Status DlModelHelper::CreateBatchFromTensors(
    std::uint32_t& batch_size,
    std::vector<tensorflow::Tensor>& input_tensor_container_down_sampled,
    std::vector<tensorflow::Tensor>& input_tensor_container_ground_truth,
    std::vector<tensorflow::Tensor>& batch_tensor_container)
{
    if (input_tensor_container_ground_truth.size() != input_tensor_container_down_sampled.size())
    {
        return tensorflow::Status(tensorflow::error::Code::INVALID_ARGUMENT, "Containers don't have same size!");
    }
    else
    {
        tensorflow::GraphDef graph;
        const auto tensor_batch_scope = tensorflow::Scope::NewRootScope();

        std::vector<tensorflow::Input> temp_tensor_container_ground_truth;
        std::vector<tensorflow::Input> temp_tensor_container_down_sampled;

        batch_size = (batch_size <= input_tensor_container_ground_truth.size())
                         ? batch_size
                         : input_tensor_container_ground_truth.size();

        auto ground_truth_tensor_container_iterator =
            std::next(input_tensor_container_ground_truth.begin(), batch_size);
        std::move(input_tensor_container_ground_truth.begin(),
                  ground_truth_tensor_container_iterator,
                  std::back_inserter(temp_tensor_container_ground_truth));
        input_tensor_container_ground_truth.erase(input_tensor_container_ground_truth.begin(),
                                                  ground_truth_tensor_container_iterator);

        auto down_sampled_tensor_container_iterator =
            std::next(input_tensor_container_down_sampled.begin(), batch_size);
        std::move(input_tensor_container_down_sampled.begin(),
                  down_sampled_tensor_container_iterator,
                  std::back_inserter(temp_tensor_container_down_sampled));
        input_tensor_container_down_sampled.erase(input_tensor_container_down_sampled.begin(),
                                                  down_sampled_tensor_container_iterator);

        tensorflow::InputList input_ground_truth_tensors(temp_tensor_container_ground_truth);
        tensorflow::ops::Stack(tensor_batch_scope.WithOpName(stack_ground_truth_container_),
                               input_ground_truth_tensors);
        tensorflow::InputList input_down_sampled_tensors(temp_tensor_container_down_sampled);
        tensorflow::ops::Stack(tensor_batch_scope.WithOpName(stack_down_sampled_container_),
                               input_down_sampled_tensors);

        TF_RETURN_IF_ERROR(tensor_batch_scope.ToGraphDef(&graph));

        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

        TF_RETURN_IF_ERROR(session->Create(graph));
        TF_RETURN_IF_ERROR(session->Run(
            {}, {stack_down_sampled_container_, stack_ground_truth_container_}, {}, &batch_tensor_container));

        return tensorflow::Status::OK();
    }
}

cv::Mat DlModelHelper::CreateImageFromTensor(tensorflow::Tensor& tensor)
{
    cv::Mat tensor_image(tensor.dim_size(1), tensor.dim_size(2), CV_32FC3, tensor.flat<float>().data());

    return tensor_image;
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

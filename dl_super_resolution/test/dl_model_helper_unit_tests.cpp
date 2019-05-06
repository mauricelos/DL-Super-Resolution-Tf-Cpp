#include "gtest/gtest.h"

#define private public
#include <dl_model_helper.h>

class DlModelHelperTest : public testing::Test
{
  public:
    DlModelHelper unit_{};

  private:
    const std::string path_to_test_image_{"dl_super_resolution/images/unit_test_image.jpeg"};
};

TEST_F(DlModelHelperTest, EndsWith_FilenameEndsWithPNGSuffix)
{
    const std::string file_name{"image_test.png"};
    const std::string file_suffix{".png"};

    EXPECT_TRUE(unit_.EndsWith(file_name, file_suffix));
}

TEST_F(DlModelHelperTest, EndsWith_FilenameEndsWithJPEGSuffix)
{
    const std::string file_name{"image_test.jpeg"};
    const std::string file_suffix{".jpeg"};

    EXPECT_TRUE(unit_.EndsWith(file_name, file_suffix));
}

TEST_F(DlModelHelperTest, EndsWith_FilenameEndsWithDifferentSuffix)
{
    const std::string file_name{"image_test.bmp"};
    const std::string file_suffix{".jpeg"};
    const std::string alternative_file_suffix{".png"};

    EXPECT_FALSE(unit_.EndsWith(file_name, file_suffix));
    EXPECT_FALSE(unit_.EndsWith(file_name, alternative_file_suffix));
}

TEST_F(DlModelHelperTest, CreateTensorFromImage_SmallTestImage)
{
    const tensorflow::TensorShape expected_dimensions{1U, 1080U, 1920U, 3U};
    const std::array<std::uint32_t, 3> tensor_dimensions{1080U, 1920U, 3U};
    std::vector<tensorflow::Tensor> unit_test_tensor_container;

    auto result = unit_.CreateTensorFromImage(path_to_test_image_, unit_test_tensor_container, tensor_dimensions);

    EXPECT_EQ(unit_test_tensor_container.front().shape(), expected_dimensions);
}

TEST_F(DlModelHelperTest, CreateBatchFromTensors_TwoTensorsSamePictureSameConfig)
{
    const tensorflow::TensorShape expected_dimensions{1U, 1U, 1080U, 1920U, 3U};
    std::uint32_t batch_size{1U};
    std::vector<tensorflow::Tensor> first_unit_test_tensor_container;
    std::vector<tensorflow::Tensor> second_unit_test_tensor_container;
    std::vector<tensorflow::Tensor> batch_tensor_container;
    auto status_first_tensor_container =
        unit_.CreateTensorFromImage(path_to_test_image_, first_unit_test_tensor_container);
    auto status_second_tensor_container =
        unit_.CreateTensorFromImage(path_to_test_image_, second_unit_test_tensor_container);

    auto result = unit_.CreateBatchFromTensors(
        batch_size, first_unit_test_tensor_container, second_unit_test_tensor_container, batch_tensor_container);

    EXPECT_EQ(batch_tensor_container.front().shape(), expected_dimensions);
    EXPECT_EQ(batch_tensor_container.back().shape(), expected_dimensions);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

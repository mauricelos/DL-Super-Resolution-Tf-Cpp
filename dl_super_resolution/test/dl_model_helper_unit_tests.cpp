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

TEST_F(DlModelHelperTest, DlModelHelper_DefaultConstructorTest)
{
    const std::uint32_t expected_input_height{1080U};
    const std::uint32_t expected_input_width{1920U};
    const std::uint32_t expected_num_required_image_channels{3U};

    EXPECT_EQ(unit_.input_height_, expected_input_height);
    EXPECT_EQ(unit_.input_width_, expected_input_width);
    EXPECT_EQ(unit_.num_required_image_channels_, expected_num_required_image_channels);
}

TEST_F(DlModelHelperTest, DlModelHelper_CustomConstructorTest)
{
    const std::uint32_t expected_input_height{900U};
    const std::uint32_t expected_input_width{1600U};
    const std::uint32_t expected_num_required_image_channels{2U};
    DlModelHelper temp_unit{expected_input_height, expected_input_width, expected_num_required_image_channels};

    EXPECT_EQ(temp_unit.input_height_, expected_input_height);
    EXPECT_EQ(temp_unit.input_width_, expected_input_width);
    EXPECT_EQ(temp_unit.num_required_image_channels_, expected_num_required_image_channels);
}

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
    const std::uint32_t expected_dimensions{4U};
    std::vector<tensorflow::Tensor> unit_test_tensor_container;

    auto result = unit_.CreateTensorFromImage(path_to_test_image_, unit_test_tensor_container);

    EXPECT_EQ(unit_test_tensor_container[0].shape().dims(), expected_dimensions);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

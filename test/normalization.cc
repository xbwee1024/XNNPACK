#include <gtest/gtest.h>

#include <xnnpack/normalization.h>

struct test_case {
  size_t num_dims;
  size_t element_size;
  std::vector<size_t> perm;
  std::vector<size_t> shape;
  std::vector<size_t> expected_normalized_shape;
  std::vector<size_t> expected_normalized_perm;
  size_t expected_normalized_dims;
  size_t expected_element_size;
};

class TransposeNormalizationTest : public ::testing::TestWithParam<test_case> {};

TEST_P(TransposeNormalizationTest, xnn_normalize_tensor_dimensions) {
  struct test_case test_data = GetParam();
  std::vector<size_t> actual_normalized_shape(test_data.num_dims);
  std::vector<size_t> actual_normalized_perm(test_data.num_dims);
  size_t actual_normalized_dims;
  size_t actual_element_size;

  xnn_normalize_tensor_dimensions(test_data.num_dims, test_data.element_size, test_data.perm.data(),
                                  test_data.shape.data(), &actual_normalized_dims, &actual_element_size,
                                  actual_normalized_perm.data(), actual_normalized_shape.data());
  EXPECT_EQ(test_data.expected_element_size, actual_element_size);
  EXPECT_EQ(test_data.expected_normalized_dims, actual_normalized_dims);

  for (size_t i = 0; i < test_data.expected_normalized_dims; ++i) {
    EXPECT_EQ(test_data.expected_normalized_shape[i], actual_normalized_shape[i]);
    EXPECT_EQ(test_data.expected_normalized_perm[i], actual_normalized_perm[i]);
  }
}

std::vector<test_case> test_cases() {
  return {
    test_case{
      .num_dims = 1,
      .element_size = 4,
      .perm = {0},
      .shape = {37},
      .expected_normalized_shape = {1},
      .expected_normalized_perm = {0},
      .expected_normalized_dims = 1,
      .expected_element_size = 37*4},
    test_case{
      .num_dims = 2,
      .element_size = 4,
      .perm = {0,1},
      .shape = {37,19},
      .expected_normalized_shape = {1},
      .expected_normalized_perm = {0},
      .expected_normalized_dims = 1,
      .expected_element_size = 37*19*4},
    test_case{
      .num_dims = 2,
      .element_size = 4,
      .perm = {1,0},
      .shape = {23,17},
      .expected_normalized_shape = {23,17},
      .expected_normalized_perm = {1,0},
      .expected_normalized_dims = 2,
      .expected_element_size = 4},
    test_case{
      .num_dims = 3,
      .element_size = 4,
      .perm = {0,1,2},
      .shape = {101,13,7},
      .expected_normalized_shape = {1},
      .expected_normalized_perm = {0},
      .expected_normalized_dims = 1,
      .expected_element_size = 101*13*7* 4},
    test_case{
      .num_dims = 3,
      .element_size = 4,
      .perm = {2,0,1},
      .shape = {101,13,7},
      .expected_normalized_shape = {101*13,7},
      .expected_normalized_perm = {1,0},
      .expected_normalized_dims = 2,
      .expected_element_size = 4},
    test_case{
      .num_dims = 3,
      .element_size = 4,
      .perm = {1,0,2},
      .shape = {101,13,7},
      .expected_normalized_shape = {101,13},
      .expected_normalized_perm = {1,0},
      .expected_normalized_dims = 2,
      .expected_element_size = 7*4},
    test_case{
      .num_dims = 3,
      .element_size = 4,
      .perm = {2,1,0},
      .shape = {101,13,7},
      .expected_normalized_shape = {101,13,7},
      .expected_normalized_perm = {2,1,0},
      .expected_normalized_dims = 3,
      .expected_element_size = 4},
    test_case{
      .num_dims = 4,
      .element_size = 1,
      .perm = {1,0,2,3},
      .shape = {101,13,7,19},
      .expected_normalized_shape = {101,13},
      .expected_normalized_perm = {1,0},
      .expected_normalized_dims = 2,
      .expected_element_size = 1*7*19},
    test_case{
      .num_dims = 4,
      .element_size = 2,
      .perm = {0,3,1,2},
      .shape = {19,31,41,7},
      .expected_normalized_shape = {19,31*41,7},
      .expected_normalized_perm = {0,2,1},
      .expected_normalized_dims = 3,
      .expected_element_size = 2},
    test_case{
      .num_dims = 5,
      .element_size = 4,
      .perm = {4,2,3,0,1},
      .shape = {19,13,31,41,7},
      .expected_normalized_shape = {19*13,31*41,7},
      .expected_normalized_perm = {2,1,0},
      .expected_normalized_dims = 3,
      .expected_element_size = 4},
    test_case{
      .num_dims = 5,
      .element_size = 2,
      .perm = {4,3,0,1,2},
      .shape = {19,13,31,41,7},
      .expected_normalized_shape = {19*13*31,41,7},
      .expected_normalized_perm = {2,1,0},
      .expected_normalized_dims = 3,
      .expected_element_size = 2},
    test_case{
      .num_dims = 5,
      .element_size = 2,
      .perm = {4,3,1,2,0},
      .shape = {19,13,31,41,7},
      .expected_normalized_shape = {19,13*31,41,7},
      .expected_normalized_perm = {3,2,1,0},
      .expected_normalized_dims = 4,
      .expected_element_size = 2},
    test_case{
      .num_dims = 6,
      .element_size = 4,
      .perm = {4,5,0,1,2,3},
      .shape = {53,19,13,31,41,7},
      .expected_normalized_shape = {53*19*13*31,41*7},
      .expected_normalized_perm = {1,0},
      .expected_normalized_dims = 2,
      .expected_element_size = 4},
    test_case{
      .num_dims = 6,
      .element_size = 4,
      .perm = {0,1,2,3,5,4},
      .shape = {53,19,13,31,41,7},
      .expected_normalized_shape = {53*19*13*31,41,7},
      .expected_normalized_perm = {0,2,1},
      .expected_normalized_dims = 3,
      .expected_element_size = 4},
    test_case{
      .num_dims = 6,
      .element_size = 4,
      .perm = {0,3,1,2,4,5},
      .shape = {53,19,13,31,41,7},
      .expected_normalized_shape = {53,19*13,31},
      .expected_normalized_perm = {0,2,1},
      .expected_normalized_dims = 3,
      .expected_element_size = 4*41*7},
    test_case{
      .num_dims = 6,
      .element_size = 4,
      .perm = {4,5,3,1,2,0},
      .shape = {53,19,13,31,41,7},
      .expected_normalized_shape = {53,19*13,31,41*7},
      .expected_normalized_perm = {3,2,1,0},
      .expected_normalized_dims = 4,
      .expected_element_size = 4}
  };
}

INSTANTIATE_TEST_SUITE_P(
    NormalizationTestCases, TransposeNormalizationTest,
    testing::ValuesIn(test_cases()));



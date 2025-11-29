#include "ops.h"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

bool verbose_g = false;

struct OpsInitializer {
  OpsInitializer() { init_ops(1); }
};
static OpsInitializer ops_initializer;

TEST(OpsTest, RmsNorm) {
  std::vector<float> o(4);
  std::vector<float> x = {1.0, 2.0, 3.0, 4.0};
  rms_norm(o, x, 1e-5f);
  float ss = 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0;
  ss /= 4.0;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  for (size_t i = 0; i < o.size(); ++i) {
    EXPECT_NEAR(o[i], ss * x[i], 1e-6);
  }
}

TEST(OpsTest, MatVecMul) {
  std::vector<float> o(2);
  std::vector<float> w = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> x = {5.0, 6.0};
  mat_vec_mul(o, w, x);
  EXPECT_NEAR(o[0], 1.0 * 5.0 + 2.0 * 6.0, 1e-6);
  EXPECT_NEAR(o[1], 3.0 * 5.0 + 4.0 * 6.0, 1e-6);
}

TEST(OpsTest, Softmax) {
  std::vector<float> x = {1.0, 2.0, 3.0, 4.0};
  softmax(x);
  float max_val = 4.0;
  float sum = expf(1.0 - max_val) + expf(2.0 - max_val) + expf(3.0 - max_val) +
              expf(4.0 - max_val);
  EXPECT_NEAR(x[0], expf(1.0 - max_val) / sum, 1e-6);
  EXPECT_NEAR(x[1], expf(2.0 - max_val) / sum, 1e-6);
  EXPECT_NEAR(x[2], expf(3.0 - max_val) / sum, 1e-6);
  EXPECT_NEAR(x[3], expf(4.0 - max_val) / sum, 1e-6);
}
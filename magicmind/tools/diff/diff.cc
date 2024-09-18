/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#include "common/macros.h"
#include "common/data.h"
#include "common/param.h"
#include "common/logger.h"
#include "third_party/half/half.h"

class DiffArg : public ArgListBase {
  DECLARE_ARG(data, (std::string))->SetDescription("Data to be evaluated for comparision.");
  DECLARE_ARG(baseline, (std::string))->SetDescription("Baseline data path for comparision.");
  DECLARE_ARG(datatype, (std::string))
      ->SetDescription("Data type for comparision.")
      ->SetAlternative({"int8", "int16", "int32", "uint8", "uint16", "uint32", "half", "float"});
  DECLARE_ARG(threshold1, (float))
      ->SetDescription("Threshold for diff 1 in float")
      ->SetDefault({"NaN"});
  DECLARE_ARG(threshold2, (float))
      ->SetDescription("Threshold for diff 2 in float")
      ->SetDefault({"NaN"});
  DECLARE_ARG(threshold3, (std::vector<float>))
      ->SetDescription("Threshold for diff 3 in float")
      ->SetDefault({"NaN", "NaN"});
  DECLARE_ARG(threshold4, (float))
      ->SetDescription("Threshold for diff 4 in float")
      ->SetDefault({"NaN"});
};

/*!
 * @struct TypeToString
 * @brief Converts Basic data type in C/C++ to string.
 * TypeToString<T>::value is the string constant for basic data type T.
 * e.g. TypeToString<float>::value is "float".
 */
template <class T>
struct TypeToString {};

// Template specialization for ataTypeToString.
#define MATCH_TYPE_STRING(TYPE, STRING) \
  template <>                           \
  struct TypeToString<TYPE> {           \
    static const std::string value;     \
  };                                    \
  const std::string TypeToString<TYPE>::value = STRING;
MATCH_TYPE_STRING(int8_t, "int8");
MATCH_TYPE_STRING(int16_t, "int16");
MATCH_TYPE_STRING(int32_t, "int32");
MATCH_TYPE_STRING(uint8_t, "uint8");
MATCH_TYPE_STRING(uint16_t, "uint16");
MATCH_TYPE_STRING(uint32_t, "uint32");
MATCH_TYPE_STRING(float, "float");
MATCH_TYPE_STRING(half_float::half, "half");

template <class T>
int CompareAllDiff(const std::string &evaluated_data,
                   const std::string &base_line,
                   float threshold1,
                   float threshold2,
                   const std::vector<float> &threshold3,
                   float threshold4) {
  CHECK_EQ(threshold3.size(), 2);
  size_t size_e = FileSize(evaluated_data);
  size_t size_b = FileSize(base_line);
  CHECK_EQ(size_e, size_b);
  std::vector<T> e(size_e / sizeof(T));
  std::vector<T> b(size_b / sizeof(T));
  CHECK_VALID(ReadDataFromFile(evaluated_data, e.data(), size_e));
  CHECK_VALID(ReadDataFromFile(base_line, b.data(), size_b));
  auto diff1 = ComputeDiff(e, b, Diff::Type1);
  auto diff2 = ComputeDiff(e, b, Diff::Type2);
  auto diff3 = ComputeDiff(e, b, Diff::Type3);
  auto diff4 = ComputeDiff(e, b, Diff::Type4);
  int ret = 0;
  SLOG(INFO) << "Diff1 : " << diff1;
  SLOG(INFO) << "Diff2 : " << diff2;
  SLOG(INFO) << "Diff3 : " << diff3;
  SLOG(INFO) << "Diff4 : " << diff4;
  if (!std::isnan(threshold1)) {
    SLOG(INFO) << "Diff1 vs. threshold1 : " << diff1[0] << " vs. " << threshold1;
    if (diff1[0] <= threshold1) {
      SLOG(INFO) << "Diff1 passed.";
    } else {
      SLOG(INFO) << "Diff1 failed.";
      ret = -1;
    }
  }
  if (!std::isnan(threshold2)) {
    SLOG(INFO) << "Diff2 vs. threshold2 : " << diff2[0] << " vs. " << threshold2;
    if (diff2[0] <= threshold2) {
      SLOG(INFO) << "Diff2 passed.";
    } else {
      SLOG(INFO) << "Diff2 failed.";
      ret = -1;
    }
  }
  if (!std::isnan(threshold3[0]) && !std::isnan(threshold3[1])) {
    SLOG(INFO) << "Diff3 vs. threshold3 : " << diff3[0] << ", " << diff3[1] << " vs. "
               << threshold3[0] << ", " << threshold3[1];
    if ((diff3[0] <= threshold3[0]) && (diff3[1] <= threshold3[1])) {
      SLOG(INFO) << "Diff3 passed.";
    } else {
      SLOG(INFO) << "Diff3 failed.";
      ret = -1;
    }
  }
  if (!std::isnan(threshold4)) {
    SLOG(INFO) << "Diff4 vs. threshold4 : " << diff4[2] / float(b.size() + 1e-6) << " vs. "
               << threshold4;
    if (diff4[2] / float(b.size() + 1e-6) <= threshold4) {
      SLOG(INFO) << "Diff4 passed.";
    } else {
      SLOG(INFO) << "Diff4 failed.";
      ret = -1;
    }
  }
  if (ret != 0) {
    SLOG(ERROR) << "Compare failed.";
  } else {
    SLOG(INFO) << "Compare successed.";
  }
  return ret;
}

int main(int argc, char *argv[]) {
  auto args = ArrangeArgs(argc, argv);
  DiffArg arg_reader;
  arg_reader.ReadIn(args);
  std::string type = Value(arg_reader.datatype());
#define CASE(t, T)                                                                         \
  if (t == TypeToString<T>::value)                                                         \
  return CompareAllDiff<T>(Value(arg_reader.data()), Value(arg_reader.baseline()),         \
                           Value(arg_reader.threshold1()), Value(arg_reader.threshold2()), \
                           Value(arg_reader.threshold3()), Value(arg_reader.threshold4()))
  CASE(type, int8_t);
  CASE(type, int16_t);
  CASE(type, int32_t);
  CASE(type, uint8_t);
  CASE(type, uint16_t);
  CASE(type, uint32_t);
  CASE(type, float);
  CASE(type, half_float::half);
  SLOG(ERROR) << "Wrong dtype";
  abort();
#undef CASE
}

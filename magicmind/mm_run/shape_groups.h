/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Function for parse/record input shapes
 *************************************************************************/
#ifndef SHAPE_GROUPS_H_
#define SHAPE_GROUPS_H_
#include <vector>
#include <map>
#include "common/json_util.h"
/*
 *  Object for record input_shapes and gain batch_size
 *  Return the max value of all shapes' very first dim as batch (could be wrong)
 */
class Shapes {
 public:
  explicit Shapes(const std::vector<std::vector<int>> &shapes);
  explicit Shapes(const std::map<std::string, std::vector<int>> &shapes);
  void Reorder(const std::vector<std::string> &names);
  int BatchSize() const;
  size_t size() const;
  std::vector<std::vector<int>> GetShapes() const;
  std::vector<int> operator[](size_t index);
  std::vector<int> operator[](const std::string &name);
  std::string DebugString() const;
  bool has_name() const;
  json11::Json ToJson() const;

 private:
  bool has_name_ = false;
  std::vector<std::vector<int>> shape_without_name_;
  std::vector<std::pair<std::string, std::vector<int>>> shape_with_name_;
};
/*
 *  Object for record input_shapes and gain batch_size
 *  Return the max values among shape groups' very first dim as batches (could be wrong)
 */
class ShapeGroups {
 public:
  ShapeGroups();
  explicit ShapeGroups(json11::Json obj);
  explicit ShapeGroups(const std::vector<std::vector<std::vector<int>>> &shapes);
  explicit ShapeGroups(const std::vector<std::map<std::string, std::vector<int>>> &shapes);

  void Reorder(const std::vector<std::string> &names);
  std::vector<int> BatchSizes() const;
  size_t size() const;
  Shapes operator[](size_t index) const;
  std::string DebugString() const;
  bool has_name() const;

 private:
  void Init(const std::vector<std::vector<std::vector<int>>> &shapes);
  void Init(const std::vector<std::map<std::string, std::vector<int>>> &shapes);

 private:
  bool has_name_ = false;
  std::vector<Shapes> shape_groups_;
};
#endif  // SHAPE_GROUPS_H_

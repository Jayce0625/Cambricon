/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Function for parse/record input shapes
 *************************************************************************/
#include <algorithm>
#include "common/macros.h"
#include "common/logger.h"
#include "mm_run/shape_groups.h"

Shapes::Shapes(const std::vector<std::vector<int>> &shapes) {
  has_name_ = false;
  shape_without_name_ = shapes;
}

Shapes::Shapes(const std::map<std::string, std::vector<int>> &shapes) {
  has_name_ = true;
  for (auto e_ : shapes) {
    shape_with_name_.push_back(e_);
  }
}

void Shapes::Reorder(const std::vector<std::string> &names) {
  CHECK_VALID(has_name_);
  CHECK_EQ(names.size(), shape_with_name_.size());
  for (auto e_ : shape_with_name_) {
    if (std::count(names.begin(), names.end(), e_.first) == 0) {
      SLOG(ERROR) << "Cant find " << e_.first << " in " << names;
      abort();
    }
  }
  std::vector<std::pair<std::string, std::vector<int>>> names_reorder;
  for (auto n : names) {
    names_reorder.push_back({n, (*this)[n]});
  }
  shape_with_name_ = names_reorder;
}

int Shapes::BatchSize() const {
  int ret = 0;
  if (has_name_) {
    for (auto e_ : shape_with_name_) {
      if (e_.second.size() > 0) {
        ret = ret < e_.second[0] ? e_.second[0] : ret;
      }
    }
  } else {
    for (auto e_ : shape_without_name_) {
      if (e_.size() > 0) {
        ret = ret < e_[0] ? e_[0] : ret;
      }
    }
  }
  return ret;
}

size_t Shapes::size() const {
  if (has_name_) {
    return shape_with_name_.size();
  } else {
    return shape_without_name_.size();
  }
}

std::vector<std::vector<int>> Shapes::GetShapes() const {
  if (has_name_) {
    std::vector<std::vector<int>> ret;
    for (auto e_ : shape_with_name_) {
      ret.push_back(e_.second);
    }
    return ret;
  } else {
    return shape_without_name_;
  }
}

std::vector<int> Shapes::operator[](size_t index) {
  if (!has_name_) {
    return shape_without_name_[index];
  } else {
    return shape_with_name_[index].second;
  }
}

std::vector<int> Shapes::operator[](const std::string &name) {
  if (has_name_) {
    for (auto e_ : shape_with_name_) {
      if (e_.first == name) {
        return e_.second;
      }
    }
  }
  return {};
}

std::string Shapes::DebugString() const {
  std::stringstream ss;
  if (has_name_) {
    ss << shape_with_name_;
  } else {
    ss << shape_without_name_;
  }
  return ss.str();
}

bool Shapes::has_name() const {
  return has_name_;
}

json11::Json Shapes::ToJson() const {
  json11::Json ret;
  if (has_name_) {
    std::map<std::string, std::vector<int>> to_json(
        {shape_with_name_.begin(), shape_with_name_.end()});
    return GetJsonObjFromValue("", to_json);
  } else {
    return GetJsonObjFromValue("", shape_without_name_);
  }
}

ShapeGroups::ShapeGroups() : ShapeGroups(std::vector<std::vector<std::vector<int>>>({{{}}})) {}

ShapeGroups::ShapeGroups(json11::Json obj) {
  int type = 0;
  CHECK_VALID(GetJsonValueFromObj(obj, "inputType", &type));
  if (type == 0) {
    std::vector<std::vector<std::vector<int>>> in_shapes;
    CHECK_VALID(GetJsonValueFromObj(obj, "inputDims", &in_shapes));
    Init(in_shapes);
  } else if (type == 1) {
    std::vector<std::map<std::string, std::vector<int>>> in_shapes;
    CHECK_VALID(GetJsonValueFromObj(obj, "inputDims", &in_shapes));
    Init(in_shapes);
  } else {
    SLOG(ERROR) << "Unsupoport inputType " << type << " for inputs!";
    abort();
  }
}

ShapeGroups::ShapeGroups(const std::vector<std::vector<std::vector<int>>> &shapes) {
  Init(shapes);
}

ShapeGroups::ShapeGroups(const std::vector<std::map<std::string, std::vector<int>>> &shapes) {
  Init(shapes);
}

void ShapeGroups::Init(const std::vector<std::vector<std::vector<int>>> &shapes) {
  has_name_ = false;
  for (auto e_ : shapes) {
    shape_groups_.push_back(Shapes(e_));
  }
}

void ShapeGroups::Init(const std::vector<std::map<std::string, std::vector<int>>> &shapes) {
  has_name_ = true;
  for (auto e_ : shapes) {
    shape_groups_.push_back(Shapes(e_));
  }
}

void ShapeGroups::Reorder(const std::vector<std::string> &names) {
  for (auto &shape : shape_groups_) {
    shape.Reorder(names);
  }
}

std::vector<int> ShapeGroups::BatchSizes() const {
  std::vector<int> ret;
  for (auto e_ : shape_groups_) {
    ret.push_back(e_.BatchSize());
  }
  return ret;
}

size_t ShapeGroups::size() const {
  return shape_groups_.size();
}

Shapes ShapeGroups::operator[](size_t index) const {
  return shape_groups_[index];
}

std::string ShapeGroups::DebugString() const {
  std::stringstream ss;
  ss << "\nInput Shape Summary:\n";
  for (size_t idx = 0; idx < size(); ++idx) {
    ss << "======== Shape group " << idx << " ========\n"
       << shape_groups_[idx].DebugString() << "\n";
  }
  ss << "========================\n";
  return ss.str();
}

bool ShapeGroups::has_name() const {
  return has_name_;
}

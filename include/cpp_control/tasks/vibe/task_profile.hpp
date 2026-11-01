#pragma once

#include <string>
#include <vector>

#include "common/deploy_manifest.hpp"

namespace cpp_control {
namespace vibe {

enum class GoalKind {
  COLOR,
  OBJECT_POSE,
  NONE,
};

/// THE cube-colour index order: the one-hot layout of `object_goal_color`, the
/// row order of the planner's palette, and what /vibe/sonic/goal_color carries. One
/// list, so the planner and the policy cannot mean different things by "4".
constexpr int NUM_CUBE_COLORS = 6;
const char* cube_color_name(int index);

/// Task semantics not expressible as tensor shapes alone.
///
/// The manifest remains authoritative for buffers and terms. This profile only
/// selects lifecycle and external command behavior once, at construction.
struct TaskProfile {
  std::string family;
  GoalKind goal = GoalKind::NONE;
  bool requires_motion = true;
  bool requires_prep = true;
  bool requires_twist = true;
  bool stand_reactive = false;
};

/// Basketball reference modes: approach hold dribble gather shot flight.
constexpr int NUM_BBALL_MODES = 6;

/// Write the one-hot of `mode` into out[dim]. Out-of-range throws: a dropped
/// or misindexed mode must not reach the actor as a silent all-zero vector.
void write_mode_onehot(int mode, int dim, float* out);

/// Family by case-insensitive substring; the rest of the id names the
/// architecture, which validate_contract checks structurally instead.
TaskProfile profile_from_task_id(const std::string& task_id);

/// Validate the logical observation contract, including deduplicated groups.
/// Throws on any task/model mismatch.
void validate_contract(const deploy::DeployManifest& manifest,
                       const TaskProfile& profile);

/// Logical q_* order represented by manifest port/group order. Unlike scanning
/// physical port names, this retains queries aliased onto another input.
std::vector<std::string> query_groups(const deploy::DeployManifest& manifest);

}  // namespace vibe
}  // namespace cpp_control

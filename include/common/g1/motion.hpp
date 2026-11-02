#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "common/g1/joint_orders.hpp"

namespace cpp_control {
namespace g1 {

/// Contact-graph nodes — THE column order of `bodywise_contact_cmd`, mirrored
/// from orcs `tasks/uolm/sensors.py::CONTACT_GRAPH_BODY_NAMES`. Resolved
/// against the demo `contact_matrix.npz` legend by NAME (the legend carries 32
/// bodies, motion.npz 37 — positional alignment between them does not exist).
constexpr int NUM_CONTACT_BODIES = 12;

inline const std::vector<std::string> CONTACT_GRAPH_BODIES = {
    "pelvis",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
};

/// Motion-topic row widths (IL-ordered, scripts/npz_motion_publisher.py):
///   MIN   jp29 | jv29 | anchor_pos3 | anchor_quat4                  (textop
///   era) FULL  ... | anchor_lin_vel3 | anchor_ang_vel3 | contact12       (the
///   planner cmds)
/// A MIN clip streams with zero twist AND zero contact — the whole adapter
/// augmentation port goes dead, so publish FULL for anything adapter-driven.
constexpr int WIRE_COLS_MIN = 2 * NUM_JOINTS + 3 + 4;
constexpr int WIRE_COLS_FULL = WIRE_COLS_MIN + 3 + 3 + NUM_CONTACT_BODIES;

/// The one reference-motion container — every tracker reads its motion through
/// this (no per-node parallel buffers). Storage is MJ-canonical joint order;
/// IL views gather through joint_orders.hpp. Sources:
///
///   from_npz()   retargeted-dataset / mjlab npz (fps, joint_pos, joint_vel,
///                body_{pos,quat,lin_vel,ang_vel}_w) + the sibling
///                contact_matrix.npz — mocke MjMotionLoader twin
///   from_wire()  motion topic rows, IL-ordered (see WIRE_COLS_*)
///   stand()      1-frame nominal reference (sonic/textop stand modes)
///   lead_in()    T-frame joint ramp between two poses, otherwise a stand
///                reference — walks the tracker onto a clip's first frame
///
/// root_*_vel_b are the planner command twists (orcs
/// robot_root_{lin,ang}_vel_cmd): reference-anchor-frame, pure clip functions —
/// heading-rebase invariant, so they never touch live robot state.
/// `bodywise_contact` is the third the planner command channel (orcs
/// bodywise_contact_cmd) — per-body robot<->OBJECT contact, so zeros is the
/// truthful value wherever there is no object (stand).
struct Motion {
  int num_frames = 0;
  int num_joints = 0;
  int num_bodies = 0;
  float fps = 50.0f;
  bool has_twist =
      false;  ///< body_*_vel_w came from the source (vs zero-filled)
  bool has_contact =
      false;  ///< bodywise_contact came from the source (vs zero-filled)
  bool has_object_goal = false;  ///< sibling object_motion / reference envelope
  std::array<float, 3> object_goal_pos = {0.0f, 0.0f, 0.0f};
  std::array<float, 4> object_goal_quat = {1.0f, 0.0f, 0.0f, 0.0f};  ///< wxyz
  bool has_modes = false;  ///< sibling modes.npz (absent = no per-frame mode)
  std::vector<int8_t> modes;            ///< (T,) index into mode_names
  std::vector<std::string> mode_names;  ///< legend carried by modes.npz

  std::vector<float> joint_pos;       ///< (T*J) MJ order
  std::vector<float> joint_vel;       ///< (T*J) MJ order
  std::vector<float> body_pos_w;      ///< (T*B*3)
  std::vector<float> body_quat_w;     ///< (T*B*4) wxyz
  std::vector<float> body_lin_vel_w;  ///< (T*B*3)
  std::vector<float> body_ang_vel_w;  ///< (T*B*3)
  std::vector<float>
      bodywise_contact;  ///< (T*K) {0,1}, CONTACT_GRAPH_BODIES order

  /// joint_perm: source column for each output joint (empty = identity;
  /// g1::MJ2IL for IL-ordered clips). Body arrays load as-is. A source with no
  /// object (for example APPROACH) sets load_companions=false so an unrelated
  /// sibling contact/object file cannot become part of its command contract.
  static Motion from_npz(const std::string& path,
                         const std::vector<int>& joint_perm = {},
                         bool load_companions = true);
  /// Install per-frame modes. Wrong length or an index outside `names` throws:
  /// a mislabelled mode must never fall back to a default (training's loader
  /// rejects the same inputs).
  void set_modes(std::vector<int8_t> frame_modes,
                 std::vector<std::string> names);
  /// Wire rows, IL-ordered, `cols` wide — WIRE_COLS_MIN or WIRE_COLS_FULL.
  static Motion from_wire(int num_frames, const float* rows,
                          int cols = WIRE_COLS_MIN);
  /// Versioned reference wire: FULL rows with independently valid twist and
  /// contact channels. Missing channels are ignored and zero-filled.
  static Motion from_wire(int num_frames, const float* rows, int cols,
                          bool has_twist, bool has_contact, float fps);
  static Motion stand(const std::vector<float>& default_angles_mj,
                      float fps = 50.0f);
  /// Smoothstep joint ramp `from`→`to`; root, twist and contact are stand's
  /// (a standing reference that only moves its joints). T clamped to >= 2.
  static Motion lead_in(const std::vector<float>& from_mj,
                        const std::vector<float>& to_mj, int num_frames,
                        float fps = 50.0f);
  /// One frame of a clip as a held reference: joints, anchor-body root pose and
  /// twist, contact and mode. Every future frame of it is that frame, which is
  /// what a clock frozen on that frame shows the policy.
  static Motion first_frame(const Motion& clip, int frame, int anchor_body = 0);

  // ── joint views ──
  const float* jp(int f) const {
    return &joint_pos[static_cast<size_t>(f) * num_joints];
  }
  const float* jv(int f) const {
    return &joint_vel[static_cast<size_t>(f) * num_joints];
  }
  /// (K,) reference per-body robot<->object contact — orcs
  /// object_bodywise_contact.
  const float* contact(int f) const {
    return &bodywise_contact[static_cast<size_t>(f) * NUM_CONTACT_BODIES];
  }
  /// Reference mode at a frame. A stand or lead-in has no modes: index 0, the
  /// mode a clip opens in, which is what the actor sees before the clip starts.
  int mode(int f) const { return has_modes ? modes[f] : 0; }
  void jp_il(int f,
             float* out) const;  ///< IL-ordered gather into out[num_joints]
  void jv_il(int f, float* out) const;

  // ── root (anchor body) views ──
  std::array<float, 3> root_pos(int f, int body = 0) const;
  std::array<float, 4> root_quat(int f, int body = 0) const;
  std::array<float, 3> root_lin_vel_b(
      int f, int body = 0) const;  ///< ref-anchor frame
  std::array<float, 3> root_ang_vel_b(
      int f, int body = 0) const;  ///< ref-anchor frame
};

/// Bounds what happens after a finite clip's last frame. The reference clamps
/// there, so without this the robot tracks the final frame forever. Returns
/// true once, `hold_s` after the clip finished. hold_s < 0 never fires (the
/// stock hold-the-last-frame behaviour).
class MotionEndGate {
 public:
  explicit MotionEndGate(double hold_s = -1.0) : hold_s_(hold_s) {}
  void reset() {
    held_s_ = 0.0;
    fired_ = false;
  }
  bool step(bool finished, double dt) {
    if (hold_s_ < 0.0 || fired_ || !finished) return false;
    if (held_s_ >= hold_s_) return fired_ = true;
    held_s_ += dt;
    return false;
  }

 private:
  double hold_s_;
  double held_s_ = 0.0;
  bool fired_ = false;
};

/// A may start a clip from a joint PD hold only once the ramp onto its first
/// frame has finished, the measured posture reached it and the robot is still
/// upright: joint PD does not balance, and a falling robot holds its posture.
/// NaN fails closed.
inline bool pd_hold_settled(float alpha, const float* q, const float* target,
                            int n, float max_err,
                            const std::array<float, 4>& imu_wxyz,
                            float max_tilt) {
  if (alpha < 1.0f) return false;
  const float up = 1.0f - 2.0f * (imu_wxyz[1] * imu_wxyz[1] +
                                  imu_wxyz[2] * imu_wxyz[2]);
  if (!(up >= std::cos(max_tilt))) return false;
  for (int i = 0; i < n; ++i)
    if (!(std::fabs(q[i] - target[i]) <= max_err)) return false;
  return true;
}

/// Playback clock + heading alignment over a Motion — the one `t` every
/// consumer samples through.
///
/// Deployment cannot teleport the robot to the clip's start pose the way
/// mjlab play does, so the reference is brought to the robot instead: at
/// engage(), a yaw-only offset `yaw(robot) * yaw(ref[start])^-1` is baked in
/// and the aligned_* views pre-apply it (sim2real `_heading_offset` pattern).
///
/// Future indices mirror mocke FutureMotionCommand.future_frames: frame + k,
/// clamped to the clip end.
class MotionClock {
 public:
  MotionClock(const Motion& motion, int anchor_body);

  /// `extra_yaw` rotates the whole reference on top of the alignment, so a
  /// planner that CHOSE a heading residual keeps it (the alignment would
  /// otherwise force it to zero). 0 = the plain heading rebase.
  void engage(const std::array<float, 4>& robot_quat, int start_frame = 0,
              float extra_yaw = 0.0f);
  /// Start from a reference held on the same pose: no re-alignment, the held
  /// clock's heading carries over.
  void continue_from(const MotionClock& held, int start_frame = 0);
  void step(double dt);  ///< advance by fps*dt motion frames

  int frame() const;
  int future_frame(int frames_ahead) const;  ///< clamped to num_frames-1
  bool finished() const { return frame() >= motion_.num_frames - 1; }

  /// Heading-aligned root views at a motion frame.
  std::array<float, 4> aligned_root_quat(int f) const;
  std::array<float, 3> aligned_root_pos(int f) const;

 private:
  const Motion& motion_;
  int anchor_body_;
  double phase_ = 0.0;  ///< playback position in motion frames
  std::array<float, 4> heading_offset_ = {1.0f, 0.0f, 0.0f, 0.0f};
  std::array<float, 3> ref_start_pos_ = {0.0f, 0.0f, 0.0f};
};

}  // namespace g1
}  // namespace cpp_control

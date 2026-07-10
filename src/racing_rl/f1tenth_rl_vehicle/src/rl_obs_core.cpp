#include "f1tenth_rl_vehicle/rl_obs_core.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace f1tenth_rl_vehicle
{

namespace
{
constexpr double kTwoPi = 2.0 * M_PI;

double clampd(double v, double lo, double hi)
{
  return std::max(lo, std::min(hi, v));
}

// Python floor-modulo (result has the sign of the divisor), for index wraparound.
int floorMod(int a, int m)
{
  int r = a % m;
  return (r < 0) ? r + m : r;
}
}  // namespace

TrackData loadTrackCsv(const std::string & path)
{
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("loadTrackCsv: cannot open " + path);
  }

  std::string line;
  std::vector<std::string> names;
  TrackData track;

  auto split = [](const std::string & s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      size_t a = tok.find_first_not_of(" \t\r\n");
      size_t b = tok.find_last_not_of(" \t\r\n");
      out.push_back(a == std::string::npos ? "" : tok.substr(a, b - a + 1));
    }
    return out;
  };

  bool have_header = false;
  int ix = -1, iy = -1, il = -1, ir = -1;
  while (std::getline(f, line)) {
    if (line.empty()) {
      continue;
    }
    size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
      continue;
    }
    if (line[first] == '#' && !have_header) {
      names = split(line.substr(first + 1));
      for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == "x_m") ix = static_cast<int>(i);
        else if (names[i] == "y_m") iy = static_cast<int>(i);
        else if (names[i] == "w_tr_left_m") il = static_cast<int>(i);
        else if (names[i] == "w_tr_right_m") ir = static_cast<int>(i);
      }
      have_header = true;
      continue;
    }
    if (line[first] == '#') {
      continue;
    }
    std::vector<std::string> cols = split(line);
    if (ix < 0 || iy < 0 || il < 0 || ir < 0) {
      throw std::runtime_error(
              "loadTrackCsv: missing required columns x_m,y_m,w_tr_left_m,w_tr_right_m in " + path);
    }
    track.x.push_back(std::stod(cols[ix]));
    track.y.push_back(std::stod(cols[iy]));
    track.w_left.push_back(std::stod(cols[il]));
    track.w_right.push_back(std::stod(cols[ir]));
  }

  if (track.x.size() < 2) {
    throw std::runtime_error("loadTrackCsv: fewer than 2 points in " + path);
  }
  return track;
}

std::array<double, 8> computeTyreSlip(
  const std::array<double, 4> & v_fwd,
  const std::array<double, 4> & v_lat,
  const std::array<double, 4> & spin_rate,
  double wheel_radius,
  const std::array<bool, 4> & active,
  double min_lat,
  double min_active_long,
  double min_passive_long)
{
  std::array<double, 8> out{};
  for (int i = 0; i < 4; ++i) {
    const double v_fwd_abs = std::abs(v_fwd[i]);
    const double slip_angle = std::atan(v_lat[i] / (v_fwd_abs + min_lat));
    const double wheel_speed = wheel_radius * spin_rate[i];
    const double min_long = active[i] ? min_active_long : min_passive_long;
    const double slip_ratio = (wheel_speed - v_fwd[i]) / (v_fwd_abs + min_long);
    out[i] = slip_ratio;
    out[i + 4] = slip_angle;
  }
  return out;
}

std::array<double, 4> computeQuasiStaticLoad(
  double ax,
  double ay,
  double h_cg,
  double lf,
  double lr,
  double track_width,
  double roll_stiffness_front,
  double gravity)
{
  const double g = std::max(gravity, 1e-6);
  lf = std::max(lf, 1e-6);
  lr = std::max(lr, 1e-6);
  const double tw = std::max(track_width, 1e-6);
  const double wheelbase = lf + lr;

  const double rear_base = 1.0 + ax * h_cg / (g * lf);
  const double front_base = 1.0 - ax * h_cg / (g * lr);
  const double lat_rear =
    (1.0 - roll_stiffness_front) * ay * h_cg * 2.0 * wheelbase / (g * tw * lf);
  const double lat_front =
    roll_stiffness_front * ay * h_cg * 2.0 * wheelbase / (g * tw * lr);

  std::array<double, 4> ratio{{
    rear_base - lat_rear,
    rear_base + lat_rear,
    front_base - lat_front,
    front_base + lat_front,
  }};
  for (double & r : ratio) {
    r = std::max(r, 0.0);
  }
  return ratio;
}

double lagAlpha(double control_dt, double t_delta)
{
  t_delta = std::max(t_delta, 1e-9);
  control_dt = std::max(control_dt, 1e-9);
  return control_dt / (t_delta + control_dt);
}

double stepFirstOrderLag(double state, double target, double alpha)
{
  alpha = clampd(alpha, 0.0, 1.0);
  return state + alpha * (target - state);
}

std::pair<double, double> mapActionToDrive(
  double throttle,
  double steering,
  double max_speed,
  double max_steer,
  double clip_actions,
  const std::string & brake_behavior)
{
  throttle = clampd(throttle, -clip_actions, clip_actions);
  steering = clampd(steering, -clip_actions, clip_actions);

  double speed;
  if (brake_behavior == "reverse") {
    speed = throttle * max_speed;
  } else {  // "stop": negative throttle commands a stop, not reverse
    speed = std::max(throttle, 0.0) * max_speed;
  }
  double steering_angle = steering * max_steer;
  return {speed, steering_angle};
}

TrackObservationBuilder::TrackObservationBuilder(
  const std::vector<double> & xs,
  const std::vector<double> & ys,
  const std::vector<double> & w_left,
  const std::vector<double> & w_right,
  const ObsConfig & cfg,
  int coarse_stride,
  int window)
: cfg_(cfg), coarse_stride_(coarse_stride), window_(window)
{
  n_ = static_cast<int>(xs.size());
  if (n_ < 2 || static_cast<int>(ys.size()) != n_) {
    throw std::runtime_error("TrackObservationBuilder: invalid centerline size");
  }
  wl_ = w_left;
  wr_ = w_right;
  nw_ = static_cast<int>(w_left.size());

  // Open-polyline cache (original centerline) used by the future-track points.
  ox_ = xs;
  oy_ = ys;
  oseg_len_.resize(n_ - 1);
  ocumlen_.assign(n_, 0.0);
  for (int i = 0; i < n_ - 1; ++i) {
    double dx = ox_[i + 1] - ox_[i];
    double dy = oy_[i + 1] - oy_[i];
    oseg_len_[i] = std::sqrt(dx * dx + dy * dy);
    ocumlen_[i + 1] = ocumlen_[i] + oseg_len_[i];
  }

  // Closed-loop cache used by Frenet projection: append the first point if the
  // centerline is not already closed (mirrors build_track_cache).
  std::vector<double> clx = xs;
  std::vector<double> cly = ys;
  double dx0 = clx.front() - clx.back();
  double dy0 = cly.front() - cly.back();
  if (std::sqrt(dx0 * dx0 + dy0 * dy0) > 1e-6) {
    clx.push_back(clx.front());
    cly.push_back(cly.front());
  }
  m_ = static_cast<int>(clx.size()) - 1;
  cx_.resize(m_);
  cy_.resize(m_);
  seg_x_.resize(m_);
  seg_y_.resize(m_);
  seg_len_.resize(m_);
  cumlen_.assign(m_, 0.0);
  for (int i = 0; i < m_; ++i) {
    cx_[i] = clx[i];
    cy_[i] = cly[i];
    seg_x_[i] = clx[i + 1] - clx[i];
    seg_y_[i] = cly[i + 1] - cly[i];
    seg_len_[i] = std::max(std::sqrt(seg_x_[i] * seg_x_[i] + seg_y_[i] * seg_y_[i]), 1e-8);
  }
  // cumlen[1:] = cumsum(seg_len[:-1]); cumlen[0] = 0.
  for (int i = 1; i < m_; ++i) {
    cumlen_[i] = cumlen_[i - 1] + seg_len_[i - 1];
  }
  total_len_ = 0.0;
  for (int i = 0; i < m_; ++i) {
    total_len_ += seg_len_[i];
  }
  for (int i = 0; i < m_; i += coarse_stride_) {
    coarse_idx_.push_back(i);
  }
}

FrenetState TrackObservationBuilder::project(double px, double py) const
{
  // Coarse nearest centerline vertex.
  int i0 = 0;
  double best_c = std::numeric_limits<double>::infinity();
  for (int idx : coarse_idx_) {
    double dx = cx_[idx] - px;
    double dy = cy_[idx] - py;
    double d2 = dx * dx + dy * dy;
    if (d2 < best_c) {
      best_c = d2;
      i0 = idx;
    }
  }

  // Fine search within +/- window segments of the coarse pick.
  FrenetState fr;
  double best_d2 = std::numeric_limits<double>::infinity();
  for (int off = -window_; off <= window_; ++off) {
    int idx = floorMod(i0 + off, m_);
    double cxi = cx_[idx], cyi = cy_[idx];
    double sx = seg_x_[idx], sy = seg_y_[idx];
    double seg_len2 = std::max(sx * sx + sy * sy, 1e-10);
    double t = ((px - cxi) * sx + (py - cyi) * sy) / seg_len2;
    t = clampd(t, 0.0, 1.0);
    double projx = cxi + t * sx;
    double projy = cyi + t * sy;
    double dx = projx - px;
    double dy = projy - py;
    double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      fr.best_idx = idx;
      fr.best_t = t;
      fr.proj_x = projx;
      fr.proj_y = projy;
    }
  }

  double bsx = seg_x_[fr.best_idx];
  double bsy = seg_y_[fr.best_idx];
  double norm = std::max(std::sqrt(bsx * bsx + bsy * bsy), 1e-8);
  fr.dir_x = bsx / norm;
  fr.dir_y = bsy / norm;
  fr.s = cumlen_[fr.best_idx] + fr.best_t * seg_len_[fr.best_idx];
  fr.L = total_len_;
  return fr;
}

LateralInfo TrackObservationBuilder::lateral(double px, double py) const
{
  const FrenetState fr = project(px, py);
  LateralInfo info;
  const double nx = -fr.dir_y;
  const double ny = fr.dir_x;
  info.ey = (px - fr.proj_x) * nx + (py - fr.proj_y) * ny;
  const int i0 = fr.best_idx;
  const int i1 = (nw_ > 0) ? (fr.best_idx + 1) % nw_ : fr.best_idx;
  info.w_left = wl_[i0] + fr.best_t * (wl_[i1] - wl_[i0]);
  info.w_right = wr_[i0] + fr.best_t * (wr_[i1] - wr_[i0]);
  info.s = fr.s;
  info.L = fr.L;
  return info;
}

std::vector<float> TrackObservationBuilder::build(
  const VehicleState & st, const OpponentState & opp) const
{
  std::vector<float> obs(static_cast<size_t>(cfg_.num_obs), 0.0f);
  const FrenetState fr = project(st.pos_x, st.pos_y);

  // [0:5] body-frame kinematics; [5:7] last action.
  obs[0] = static_cast<float>(st.vx * cfg_.lin_vel_scale);
  obs[1] = static_cast<float>(st.vy * cfg_.lin_vel_scale);
  obs[2] = static_cast<float>(st.wz * cfg_.ang_vel_scale);
  obs[3] = static_cast<float>(st.ax * cfg_.lin_acc_scale);
  obs[4] = static_cast<float>(st.ay * cfg_.lin_acc_scale);
  obs[5] = static_cast<float>(st.last_throttle);
  obs[6] = static_cast<float>(st.last_steer);

  // [7:9] track progress (cos, sin) of Frenet s/L.
  double track_len = std::max(fr.L, 1e-6);
  double angle = kTwoPi * (fr.s / track_len);
  obs[7] = static_cast<float>(std::cos(angle));
  obs[8] = static_cast<float>(std::sin(angle));

  // [9] centerline heading error (wrapped).
  double track_angle = std::atan2(fr.dir_y, fr.dir_x);
  double theta_err = st.yaw - track_angle;
  theta_err = std::atan2(std::sin(theta_err), std::cos(theta_err));
  obs[9] = static_cast<float>(theta_err);

  // [10] signed lateral error ey; [11] wall-contact flag.
  double nx = -fr.dir_y;
  double ny = fr.dir_x;
  double ey = (st.pos_x - fr.proj_x) * nx + (st.pos_y - fr.proj_y) * ny;
  obs[10] = static_cast<float>(ey);

  int i0 = fr.best_idx;
  int i1 = (nw_ > 0) ? (fr.best_idx + 1) % nw_ : fr.best_idx;
  double w_l = wl_[i0] + fr.best_t * (wl_[i1] - wl_[i0]);
  double w_r = wr_[i0] + fr.best_t * (wr_[i1] - wr_[i0]);
  double d_left = w_l - ey;
  double d_right = w_r + ey;
  double boundary_dist = std::min(d_left, d_right);
  obs[11] = (boundary_dist < cfg_.contact_margin_m) ? 1.0f : 0.0f;

  // [12:372] future track points: center/left/right x samples x 2D in ego frame.
  const int samples = cfg_.future_track_num_points;
  const double total = std::max(fr.L, 1e-6);
  const double speed = std::sqrt(st.vx * st.vx + st.vy * st.vy);
  const double lookahead =
    std::max(speed * cfg_.future_track_horizon_s, cfg_.future_track_min_lookahead_m);
  // future_track_width is deprecated: corridor edges use per-vertex CSV widths.
  const double cos_y = std::cos(st.yaw);
  const double sin_y = std::sin(st.yaw);

  const int base_center = 12;
  const int base_left = base_center + samples * 2;
  const int base_right = base_left + samples * 2;

  for (int k = 1; k <= samples; ++k) {
    double step = static_cast<double>(k) / samples;
    double s_target = std::fmod(fr.s + lookahead * step, total);
    if (s_target < 0.0) {
      s_target += total;
    }
    // searchsorted(ocumlen, s_target, right=True) - 1, clamped to [0, n-2].
    int seg_idx =
      static_cast<int>(std::upper_bound(ocumlen_.begin(), ocumlen_.end(), s_target) -
      ocumlen_.begin()) - 1;
    seg_idx = std::max(0, std::min(seg_idx, n_ - 2));

    double p0x = ox_[seg_idx], p0y = oy_[seg_idx];
    double p1x = ox_[seg_idx + 1], p1y = oy_[seg_idx + 1];
    double seg_len_sel = std::max(oseg_len_[seg_idx], 1e-8);
    double alpha = (s_target - ocumlen_[seg_idx]) / seg_len_sel;

    double cx = p0x + alpha * (p1x - p0x);
    double cy = p0y + alpha * (p1y - p0y);
    double tx = (p1x - p0x) / seg_len_sel;
    double ty = (p1y - p0y) / seg_len_sel;
    double n2x = -ty;
    double n2y = tx;

    const int wi0 = seg_idx;
    const int wi1 = std::min(seg_idx + 1, nw_ - 1);
    const double w_l = wl_[wi0] + alpha * (wl_[wi1] - wl_[wi0]);
    const double w_r = wr_[wi0] + alpha * (wr_[wi1] - wr_[wi0]);
    double lx = cx + w_l * n2x;
    double ly = cy + w_l * n2y;
    double rx = cx - w_r * n2x;
    double ry = cy - w_r * n2y;

    auto to_ego = [&](double wx, double wy, double & ox, double & oy) {
      double dx = wx - st.pos_x;
      double dy = wy - st.pos_y;
      ox = cos_y * dx + sin_y * dy;
      oy = -sin_y * dx + cos_y * dy;
    };

    double ex, ey2;
    int slot = (k - 1) * 2;
    to_ego(cx, cy, ex, ey2);
    obs[base_center + slot] = static_cast<float>(ex);
    obs[base_center + slot + 1] = static_cast<float>(ey2);
    to_ego(lx, ly, ex, ey2);
    obs[base_left + slot] = static_cast<float>(ex);
    obs[base_left + slot + 1] = static_cast<float>(ey2);
    to_ego(rx, ry, ex, ey2);
    obs[base_right + slot] = static_cast<float>(ex);
    obs[base_right + slot + 1] = static_cast<float>(ey2);
  }

  // [372:380] tyre slip.
  const int slip_base = 372;
  for (int i = 0; i < 8 && slip_base + i < cfg_.num_obs; ++i) {
    obs[slip_base + i] = static_cast<float>(st.tyre_slip[i]);
  }

  // [380:384] per-wheel normal-load ratio Fz / Fz_static.
  const int load_base = 380;
  for (int i = 0; i < 4 && load_base + i < cfg_.num_obs; ++i) {
    obs[load_base + i] = static_cast<float>(st.tyre_load[i]);
  }

  // [384:390] opponent-relative block (port of f1tenth_env obs_opponent). Pure
  // relative features; the caller zeros the block when the opponent is masked out.
  if (cfg_.enable_opponent_obs) {
    const int opp_base = 384;
    if (opp_base + cfg_.opponent_obs_dim <= cfg_.num_obs) {
      if (cfg_.zero_opponent_obs) {
        for (int i = 0; i < cfg_.opponent_obs_dim; ++i) {
          obs[opp_base + i] = 0.0f;
        }
      } else {
        // Ego world-frame velocity from body velocity + yaw (obs_opponent rotates
        // the relative world velocity into the ego frame).
        const double ego_vx_w = cos_y * st.vx - sin_y * st.vy;
        const double ego_vy_w = sin_y * st.vx + cos_y * st.vy;

        const double dx = opp.pos_x - st.pos_x;
        const double dy = opp.pos_y - st.pos_y;
        const double rel_x = cos_y * dx + sin_y * dy;
        const double rel_y = -sin_y * dx + cos_y * dy;

        const double dvx = opp.vx - ego_vx_w;
        const double dvy = opp.vy - ego_vy_w;
        const double rel_vx = cos_y * dvx + sin_y * dvy;
        const double rel_vy = -sin_y * dvx + cos_y * dvy;

        const LateralInfo opp_lat = lateral(opp.pos_x, opp.pos_y);
        const double L = std::max(fr.L, 1e-6);
        double gap = opp_lat.s - fr.s;
        const double half = 0.5 * L;
        if (gap > half) {
          gap -= L;
        } else if (gap < -half) {
          gap += L;
        }
        const double gap_norm = gap / std::max(half, 1e-6);

        obs[opp_base + 0] = static_cast<float>(rel_x);
        obs[opp_base + 1] = static_cast<float>(rel_y);
        obs[opp_base + 2] = static_cast<float>(rel_vx);
        obs[opp_base + 3] = static_cast<float>(rel_vy);
        obs[opp_base + 4] = static_cast<float>(gap_norm);
        obs[opp_base + 5] = static_cast<float>(opp_lat.ey);
      }
    }
  }

  if (cfg_.clip_obs > 0.0) {
    float c = static_cast<float>(cfg_.clip_obs);
    for (auto & v : obs) {
      v = std::max(-c, std::min(c, v));
    }
  }
  return obs;
}

// --- OpponentDetector --------------------------------------------------------

OpponentDetector::OpponentDetector(
  const TrackObservationBuilder & track, const DetectorConfig & cfg)
: track_(track), cfg_(cfg)
{
}

std::vector<Cluster> OpponentDetector::cluster(const std::vector<ScanPoint> & beams) const
{
  std::vector<Cluster> out;
  const int n = static_cast<int>(beams.size());
  int i = 0;
  while (i < n) {
    if (!beams[i].valid) {
      ++i;
      continue;
    }
    const int start = i;
    double sx = beams[i].x;
    double sy = beams[i].y;
    double prevx = beams[i].x;
    double prevy = beams[i].y;
    int count = 1;
    int last = i;
    int j = i + 1;
    while (j < n && beams[j].valid) {
      const double dx = beams[j].x - prevx;
      const double dy = beams[j].y - prevy;
      if (std::sqrt(dx * dx + dy * dy) > cfg_.cluster_gap_m) {
        break;
      }
      sx += beams[j].x;
      sy += beams[j].y;
      prevx = beams[j].x;
      prevy = beams[j].y;
      ++count;
      last = j;
      ++j;
    }
    Cluster c;
    c.count = count;
    c.cx = sx / count;
    c.cy = sy / count;
    c.i0 = start;
    c.i1 = last;
    const double ex = beams[last].x - beams[start].x;
    const double ey = beams[last].y - beams[start].y;
    c.extent = std::sqrt(ex * ex + ey * ey);
    out.push_back(c);
    i = j;
  }
  return out;
}

bool OpponentDetector::inCorridor(const Cluster & c) const
{
  if (c.extent < cfg_.min_opponent_size_m || c.extent > cfg_.max_opponent_size_m) {
    return false;
  }
  // Optional tighter lower bound on physical extent (<= 0 disables).
  if (cfg_.min_extent_m > 0.0 && c.extent < cfg_.min_extent_m) {
    return false;
  }
  const LateralInfo lat = track_.lateral(c.cx, c.cy);
  const double margin = cfg_.boundary_margin_m;
  if (lat.ey > lat.w_left - margin) {
    return false;
  }
  if (lat.ey < -lat.w_right + margin) {
    return false;
  }
  return true;
}

double OpponentDetector::clusterNearEdgeDist(
  const std::vector<ScanPoint> & beams, const Cluster & c,
  double ego_x, double ego_y) const
{
  const int n = static_cast<int>(beams.size());
  double r_near = std::numeric_limits<double>::infinity();
  for (int i = c.i0; i <= c.i1 && i < n; ++i) {
    if (beams[i].valid) {
      const double dx = beams[i].x - ego_x;
      const double dy = beams[i].y - ego_y;
      r_near = std::min(r_near, std::sqrt(dx * dx + dy * dy));
    }
  }
  return r_near;
}

bool OpponentDetector::passesBeamGates(const Cluster & c, double scan_angle_increment) const
{
  // Reject speckle / single-beam wall fragments (<= 0 disables).
  if (cfg_.min_cluster_beams > 0 && c.count < cfg_.min_cluster_beams) {
    return false;
  }
  // Require minimum angular width in bearing space (<= 0 disables).
  if (cfg_.min_visible_angle_deg > 0.0) {
    const double inc = (scan_angle_increment > 0.0) ?
      scan_angle_increment : cfg_.scan_angle_increment;
    if (inc > 0.0) {
      const double span_rad = static_cast<double>(c.i1 - c.i0 + 1) * inc;
      const double span_deg = span_rad * 180.0 / M_PI;
      if (span_deg < cfg_.min_visible_angle_deg) {
        return false;
      }
    }
  }
  return true;
}

bool OpponentDetector::standsOut(
  const std::vector<ScanPoint> & beams, const Cluster & c,
  double ego_x, double ego_y) const
{
  const int n = static_cast<int>(beams.size());
  auto rng = [&](int i) {
    const double dx = beams[i].x - ego_x;
    const double dy = beams[i].y - ego_y;
    return std::sqrt(dx * dx + dy * dy);
  };

  // Near edge of the cluster (closest return to the ego).
  double r_near = clusterNearEdgeDist(beams, c, ego_x, ego_y);
  if (!std::isfinite(r_near)) {
    return false;
  }
  // Reject bumper-range phantoms (<= 0 disables).
  if (cfg_.min_ego_dist_m > 0.0 && r_near < cfg_.min_ego_dist_m) {
    return false;
  }
  if (cfg_.max_range_m > 0.0 && r_near > cfg_.max_range_m) {
    return false;
  }
  if (cfg_.foreground_jump_m <= 0.0) {
    return true;
  }

  // A flank is "background" when the immediately adjacent beam is off the array,
  // an empty return (free space), or a return well behind the cluster's near edge.
  const int lp = c.i0 - 1;
  const int rp = c.i1 + 1;
  const bool left_bg =
    lp < 0 || !beams[lp].valid || (rng(lp) - r_near) > cfg_.foreground_jump_m;
  const bool right_bg =
    rp >= n || !beams[rp].valid || (rng(rp) - r_near) > cfg_.foreground_jump_m;
  return left_bg && right_bg;
}

DetectionResult OpponentDetector::update(
  const std::vector<ScanPoint> & beams, double ego_x, double ego_y, double stamp,
  double scan_angle_increment)
{
  const std::vector<Cluster> clusters = cluster(beams);

  std::vector<Cluster> cand;
  for (const auto & c : clusters) {
    if (!passesBeamGates(c, scan_angle_increment)) {
      continue;
    }
    if (inCorridor(c) && standsOut(beams, c, ego_x, ego_y)) {
      cand.push_back(c);
    }
  }

  DetectionResult res;
  if (cand.empty()) {
    has_track_ = false;
    age_ = 0;
    return res;
  }

  // Reference point for nearest-neighbour selection: predicted track position when
  // tracking, else the ego (pick the nearest in-corridor obstacle to initialise).
  double px_ref;
  double py_ref;
  const bool tracking = has_track_;
  if (tracking) {
    double dt_pred = stamp - t_stamp_;
    if (dt_pred < 0.0) {
      dt_pred = 0.0;
    }
    px_ref = tx_ + tvx_ * dt_pred;
    py_ref = ty_ + tvy_ * dt_pred;
  } else {
    px_ref = ego_x;
    py_ref = ego_y;
  }

  int best = 0;
  double best_d2 = std::numeric_limits<double>::infinity();
  for (int i = 0; i < static_cast<int>(cand.size()); ++i) {
    const double dx = cand[i].cx - px_ref;
    const double dy = cand[i].cy - py_ref;
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      best = i;
    }
  }
  const Cluster & pick = cand[best];

  if (tracking && std::sqrt(best_d2) <= cfg_.max_assoc_dist_m) {
    const double dt = stamp - t_stamp_;
    if (dt > 1e-4) {
      const double raw_vx = (pick.cx - tx_) / dt;
      const double raw_vy = (pick.cy - ty_) / dt;
      const double a = cfg_.vel_alpha;
      tvx_ = a * raw_vx + (1.0 - a) * tvx_;
      tvy_ = a * raw_vy + (1.0 - a) * tvy_;
    }
    tx_ = pick.cx;
    ty_ = pick.cy;
    t_stamp_ = stamp;
    age_ += 1;
  } else {
    // (Re)initialise the track on the chosen candidate.
    tx_ = pick.cx;
    ty_ = pick.cy;
    tvx_ = 0.0;
    tvy_ = 0.0;
    t_stamp_ = stamp;
    has_track_ = true;
    age_ = 1;
  }

  // Persistence-based confirmation (works for a stationary opponent). Motion only
  // fast-tracks confirmation; it never rejects a slow / static track.
  const double speed = std::sqrt(tvx_ * tvx_ + tvy_ * tvy_);
  bool confirmed = age_ >= cfg_.min_track_age_frames;
  if (!confirmed && cfg_.min_speed_mps > 0.0 && speed >= cfg_.min_speed_mps) {
    confirmed = true;
  }
  if (confirmed) {
    res.present = true;
    res.pos_x = tx_;
    res.pos_y = ty_;
    res.vx = tvx_;
    res.vy = tvy_;
  }
  return res;
}

}  // namespace f1tenth_rl_vehicle

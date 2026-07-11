"""Torch quaternion and geometry helpers using the (w, x, y, z) convention."""

from __future__ import annotations

import torch

from . import runtime as rt


def inv_quat(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:].neg_()
    return out


def transform_by_quat(v: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    q_w, q_x, q_y, q_z = quat[..., :1], quat[..., 1:2], quat[..., 2:3], quat[..., 3:]
    q_ww, q_wx, q_wy, q_wz = q_w * q_w, q_w * q_x, q_w * q_y, q_w * q_z
    q_xx, q_xy, q_xz = q_x * q_x, q_x * q_y, q_x * q_z
    q_yy, q_yz = q_y * q_y, q_y * q_z
    q_zz = q_z ** 2

    vs = v / (q_ww + q_xx + q_yy + q_zz)
    v_x, v_y, v_z = vs[..., :1], vs[..., 1:2], vs[..., 2:]

    u_x = (
        v_x * (q_xx + q_ww - q_yy - q_zz)
        + v_y * (2.0 * q_xy - 2.0 * q_wz)
        + v_z * (2.0 * q_xz + 2.0 * q_wy)
    )
    u_y = (
        v_x * (2.0 * q_wz + 2.0 * q_xy)
        + v_y * (q_ww - q_xx + q_yy - q_zz)
        + v_z * (2.0 * q_yz - 2.0 * q_wx)
    )
    u_z = (
        v_x * (2.0 * q_xz - 2.0 * q_wy)
        + v_y * (2.0 * q_wx + 2.0 * q_yz)
        + v_z * (q_ww - q_xx - q_yy + q_zz)
    )
    return torch.cat([u_x, u_y, u_z], dim=-1)


def inv_transform_by_quat(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    return transform_by_quat(pos, inv_quat(quat))


def xyz_to_quat(
    xyz: torch.Tensor, rpy: bool = False, degrees: bool = False
) -> torch.Tensor:
    if degrees:
        xyz = torch.deg2rad(xyz)
    roll2, pitch2, yaw2 = (0.5 * xyz).unbind(-1)
    cosr, sinr = roll2.cos(), roll2.sin()
    cosp, sinp = pitch2.cos(), pitch2.sin()
    cosy, siny = yaw2.cos(), yaw2.sin()
    sign = 1.0 if rpy else -1.0

    out = torch.empty(xyz.shape[:-1] + (4,), dtype=xyz.dtype, device=xyz.device)
    out[..., 0] = cosr * cosp * cosy + sign * sinr * sinp * siny
    out[..., 1] = sinr * cosp * cosy - sign * cosr * sinp * siny
    out[..., 2] = cosr * sinp * cosy + sign * sinr * cosp * siny
    out[..., 3] = cosr * cosp * siny - sign * sinr * sinp * cosy
    return out


def quat_to_xyz(
    quat: torch.Tensor, rpy: bool = False, degrees: bool = False
) -> torch.Tensor:
    q_w, q_x, q_y, q_z = quat[..., :1], quat[..., 1:2], quat[..., 2:3], quat[..., 3:]
    q_ww, q_wx, q_wy, q_wz = q_w * q_w, q_w * q_x, q_w * q_y, q_w * q_z
    q_xx, q_xy, q_xz = q_x * q_x, q_x * q_y, q_x * q_z
    q_yy, q_yz = q_y * q_y, q_y * q_z
    q_zz = q_z ** 2

    if rpy:
        sinp = q_wy - q_xz
        sinrcosp = q_wx + q_yz
        sinycosp = q_wz + q_xy
    else:
        sinp = q_xz + q_wy
        sinrcosp = q_wx - q_yz
        sinycosp = q_wz - q_xy
    cosrcosp = (q_ww - q_xx - q_yy + q_zz) / 2
    cosycosp = (q_ww + q_xx - q_yy - q_zz) / 2
    cosp = torch.sqrt(cosycosp ** 2 + sinycosp ** 2)

    x = torch.atan2(sinrcosp, cosrcosp)
    y = torch.atan2(sinp, cosp)
    z = torch.atan2(sinycosp, cosycosp)

    # Special treatment of nearly singular rotations (gimbal lock).
    cosp_mask = cosp < rt.EPS
    if rpy:
        sinycosp_alt = q_wz - q_xy
    else:
        sinycosp_alt = q_wz + q_xy
    cospcosy_alt = (q_ww - q_xx + q_yy - q_zz) / 2
    x = x.masked_fill(cosp_mask, 0.0)
    z = torch.where(cosp_mask, torch.atan2(sinycosp_alt, cospcosy_alt), z)

    xyz = torch.cat([x, y, z], dim=-1)
    if degrees:
        xyz = torch.rad2deg(xyz)
    return xyz

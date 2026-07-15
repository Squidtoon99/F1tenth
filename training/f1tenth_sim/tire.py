import warp as wp

from .params import SimParams


@wp.struct
class TireForce:
    peak: wp.float32
    fx: wp.float32
    fy: wp.float32


@wp.func
def magic_formula(
    slip: wp.float32,
    b: wp.float32,
    c: wp.float32,
    e: wp.float32,
) -> wp.float32:
    bx = b * slip
    return wp.sin(c * wp.atan(bx - e * (bx - wp.atan(bx))))


@wp.func
def combined_pacejka(
    kappa: wp.float32,
    alpha: wp.float32,
    fz: wp.float32,
    mu: wp.float32,
    fz0: wp.float32,
    params: SimParams,
) -> TireForce:
    out = TireForce()
    normal_load = wp.max(fz, 0.0)
    relative_load = normal_load / wp.max(fz0, 1.0e-6) - 1.0
    load_mu = mu * (1.0 - params.tire_load_sens * relative_load)
    out.peak = wp.max(load_mu, 1.0e-4) * normal_load
    fx0 = out.peak * magic_formula(
        kappa, params.tire_b_long, params.tire_c_long, params.tire_e_long
    )
    fy0 = out.peak * magic_formula(
        alpha, params.tire_b_lat, params.tire_c_lat, params.tire_e_lat
    )
    peak_safe = wp.max(out.peak, 1.0e-6)
    combined = wp.sqrt(
        (fx0 / peak_safe) * (fx0 / peak_safe)
        + (fy0 / peak_safe) * (fy0 / peak_safe)
    )
    scale = 1.0 / wp.max(combined, 1.0)
    out.fx = fx0 * scale
    out.fy = fy0 * scale
    return out

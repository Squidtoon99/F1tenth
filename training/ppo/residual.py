RESIDUAL_ALPHA_END_FRAC = 0.25


def residual_alpha(
    transitions,
    total_transitions,
    end_frac=RESIDUAL_ALPHA_END_FRAC,
):
    horizon = float(end_frac) * float(total_transitions)
    if horizon <= 0.0:
        return 0.0
    return max(0.0, 1.0 - float(transitions) / horizon)


def mix_residual_actions(policy_action, base_action, alpha, clip=1.0):
    mixed = float(alpha) * base_action + policy_action
    limit = float(clip)
    return mixed.clamp(-limit, limit)

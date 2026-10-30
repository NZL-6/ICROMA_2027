"""Uniform acceptance, independent of solver status labels."""
import math

GAP_LIMIT = 0.001  # 0.1 percent

def relative_gap(upper, lower):
    assert math.isfinite(upper) and math.isfinite(lower)
    assert lower <= upper + 1e-6
    return max(0., upper - lower) / max(1., abs(upper))

def accepted(upper, lower):
    return relative_gap(upper, lower) < GAP_LIMIT

def full_needs_refinement(full, restricted):
    return min(r['objective'] for r in restricted) < full['objective']

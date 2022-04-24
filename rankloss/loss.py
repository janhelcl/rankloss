"""
"""
import jax
from jax import numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from .fast_soft_sort import jax_ops


def naive_approx_ranks(preds, alpha=100):
    """
    Computes approximate ranks
    """
    s_x, s_y = preds, preds
    s_xy = s_x.reshape(-1, 1) - s_y
    pairs = jnp.exp(-alpha*s_xy) / (1 + jnp.exp(-alpha*s_xy))
    return .5 + jnp.sum(pairs, axis=1)
    
    
def approx_ranks(preds):
    """
    """
    return jax_ops.soft_rank(preds)

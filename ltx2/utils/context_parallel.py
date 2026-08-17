"""Thin re-export: the da3 inference stack and the vigeo model must share ONE
context-parallel state (fastvideo.utils.context_parallel), otherwise CP inits
in one module while the model checks the other."""
from fastvideo.utils.context_parallel import *  # noqa: F401,F403

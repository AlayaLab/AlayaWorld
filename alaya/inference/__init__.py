"""da3 streaming inference pipeline (the flash release), living beside the vigeo
training/inference stack. Runs on the shared ltx2/alaya modules; da3-divergent
warp variants are kept locally in warp_da3.py so vigeo logic stays untouched.

    engine = InferenceEngine(cfg); engine.setup()
    pipe   = FlashAlayaPipeline(engine)
    cache  = pipe.initialize_cache(...) -> [generate -> finalize] x rounds -> decode
"""
from alaya.inference.cache import RolloutCache
from alaya.inference.engine import InferenceEngine
from alaya.inference.pipeline import FlashAlayaPipeline

__all__ = ["FlashAlayaPipeline", "InferenceEngine", "RolloutCache"]

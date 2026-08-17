import time
import hashlib
import json
import os
import random
import re
import signal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

import pickle
from decord import VideoReader, cpu
from torchvision import transforms
import glob


SCENE_BALANCED_PREFIX_BY_DATASET = {
    'mugen_v3': 'A handheld panoramic camera footage with natural shaking, lens distortion, and first-person walking perspective. ',
}


def _default_normalized_intrinsic():
    return np.array(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _strip_camera_blocks(text):
    if not isinstance(text, str) or '<camera' not in text.lower():
        return text
    text = re.sub(r'<camera\b[^>]*>.*?</camera>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r'\s+', ' ', text).strip()


def _score_camera_subwindows(cam_c2w_full, n_frames, fps_ratio, video_length,
                              window_size, smooth_kernel=7, angle_threshold_deg=15.0):
    """
    Score every candidate sub-window of a video and prefer clips containing turns or look-backs.

    Scoring uses the yaw change and translation magnitude derived from the camera extrinsics:
      1. compute per-frame yaw change and translation (smoothed to remove jitter)
      2. score each sub-window:
         - reversing turn (left then right): 6+
         - look-back (cumulative yaw change > 120 deg): 3+
         - clear turn (cumulative yaw > 30 deg): 2+
         - steady motion in one direction (translation, no turn): 1+
         - static (no translation, no turn): 0.5

    Args:
        cam_c2w_full: [N, 4, 4] camera-to-world extrinsics for the whole video
        n_frames: sub-window length in frames at the target fps
        fps_ratio: video_fps / target_fps
        video_length: total number of frames in the video
        window_size: sub-window length in source frames = int(n_frames * fps_ratio)
        smooth_kernel: smoothing kernel size (removes jitter)
        angle_threshold_deg: turn detection threshold in degrees

    Returns:
        list of (start_frame, score) sorted by descending score
    """
    from scipy.spatial.transform import Rotation as R

    N = len(cam_c2w_full)
    if N < 2:
        return [(0, 0.0)]

    yaw_angles = []
    translations = []
    for i in range(N):
        rot = cam_c2w_full[i, :3, :3]
        trans = cam_c2w_full[i, :3, 3]
        try:
            euler = R.from_matrix(rot).as_euler('yxz', degrees=True)
            yaw_angles.append(euler[0])
        except:
            yaw_angles.append(0.0)
        translations.append(trans)

    yaw_angles = np.array(yaw_angles, dtype=np.float64)
    translations = np.array(translations, dtype=np.float64)  # [N, 3]

    if smooth_kernel > 1 and len(yaw_angles) > smooth_kernel:
        kernel = np.ones(smooth_kernel) / smooth_kernel
        yaw_smoothed = np.convolve(yaw_angles, kernel, mode='same')
    else:
        yaw_smoothed = yaw_angles

    yaw_diff = np.diff(yaw_smoothed)
    trans_diff = np.linalg.norm(np.diff(translations, axis=0), axis=1)  # [N-1] per-frame translation distance

    if smooth_kernel > 1 and len(trans_diff) > smooth_kernel:
        trans_diff_smoothed = np.convolve(trans_diff, kernel, mode='same')
    else:
        trans_diff_smoothed = trans_diff

    max_start = video_length - window_size
    if max_start <= 0:
        return [(0, 1.0)]

    step = max(1, max_start // 200)
    candidates = list(range(0, max_start + 1, step))
    if candidates[-1] != max_start:
        candidates.append(max_start)  # always include the tail of the video

    scored = []
    for start in candidates:
        end = min(start + window_size, N - 1)
        if end <= start:
            scored.append((start, 0.0))
            continue

        seg_yaw = yaw_diff[start:end]
        seg_trans = trans_diff_smoothed[start:end]
        if len(seg_yaw) == 0:
            scored.append((start, 0.0))
            continue

        total_yaw_change = np.sum(np.abs(seg_yaw))
        total_translation = np.sum(seg_trans)
        avg_trans_per_frame = total_translation / len(seg_trans) if len(seg_trans) > 0 else 0.0

        has_translation = avg_trans_per_frame > 0.001

        significant = seg_yaw[np.abs(seg_yaw) > angle_threshold_deg / 10]

        if len(significant) < 2:
            if has_translation:
                score = 1.0 + min(total_translation * 10, 0.9)  # 1.0 ~ 1.9
            else:
                score = 0.5
        else:
            signs = np.sign(significant)
            sign_changes = np.diff(signs)
            n_reversals = np.count_nonzero(sign_changes)

            if n_reversals >= 2:
                score = 6.0 + min(n_reversals, 5) * 1.0 + total_yaw_change / 180.0
            elif n_reversals == 1 and total_yaw_change > 60:
                score = 3.0 + total_yaw_change / 180.0
            elif total_yaw_change > angle_threshold_deg * 2:
                score = 2.0 + total_yaw_change / 180.0
            else:
                if has_translation:
                    score = 1.5 + total_yaw_change / 360.0
                else:
                    score = 0.8 + total_yaw_change / 360.0

        scored.append((start, score))

    max_start = video_length - window_size
    if max_start <= 0:
        return [(0, scored[0][1] if scored else 0.0)]

    half_win = max(window_size // 2, 1)
    n_buckets = max(2, max_start // half_win + 1)
    bucket_size = max(1, (max_start + 1) // n_buckets)

    buckets = {}
    for start, sc in scored:
        if start > max_start:
            continue
        bid = start // bucket_size
        if bid not in buckets or sc > buckets[bid][1]:
            buckets[bid] = (start, sc)

    last_bid = max_start // bucket_size
    if last_bid not in buckets:
        best_last = (max_start, 0.0)
        for s, sc in scored:
            if s <= max_start and abs(s - max_start) < abs(best_last[0] - max_start):
                best_last = (s, sc)
            elif abs(s - max_start) <= bucket_size:
                if sc > best_last[1]:
                    best_last = (s, sc)
        buckets[last_bid] = best_last
    else:
        cur_start, cur_score = buckets[last_bid]
        for s, sc in scored:
            if s > cur_start and s <= max_start and sc >= cur_score - 0.01:
                buckets[last_bid] = (s, sc)
                cur_start, cur_score = s, sc

    result = list(buckets.values())
    return result if result else [(0, 0.0)]


def normalize_intrinsic(intrinsic, width, height):
    """Normalize camera intrinsics into the [0, 1] range."""
    intrinsic = intrinsic.copy()
    intrinsic[0, 0] /= width   # fx
    intrinsic[1, 1] /= height  # fy
    intrinsic[0, 2] /= width   # cx
    intrinsic[1, 2] /= height  # cy
    return intrinsic


def normalize_cam_c2w(cam_c2w, mode="max", post_scale=1.0):
    """Normalize camera extrinsics: align to the first frame and rescale translations.

    1. Left-multiply every frame by inverse(c2w[0]) so the first frame becomes identity.
    2. Rescale translations according to mode:
       - "max":   divide by max(||t_i||) so max(||t||) is about 1.0
       - "relic": divide by mean(||dP_c||), the mean per-frame step in camera coordinates
       - "none":  only align the first frame; no rescaling and no post_scale
    3. post_scale: an extra global constant applied after "max"/"relic" scaling (default 1.0)
       In relic mode |t|_max can be as large as the window length, which would overflow the
       positional encoding, so a dataset-wide constant keeps |t|_max in a safe range.
       so that |t|_max after RELIC x scale stays around 10 (safe for PRoPE in bf16).

    Args:
        cam_c2w: [N, 4, 4] camera-to-world matrices
        mode: normalization mode ("max", "relic", "none")
        post_scale: extra global constant applied after "max"/"relic" scaling; ignored when mode="none"

    Returns:
        [N, 4, 4] processed extrinsics
    """
    eps = 1e-10
    c2w0_inv = np.linalg.inv(cam_c2w[0])
    aligned = np.array([c2w0_inv @ c for c in cam_c2w])

    if mode == "max":
        translations = aligned[:, :3, 3]
        max_norm = np.linalg.norm(translations, axis=1).max()
        if max_norm > eps:
            aligned[:, :3, 3] /= max_norm
    elif mode == "relic":
        R_c2w = aligned[:, :3, :3]          # (N, 3, 3)
        P_w = aligned[:, :3, 3]             # (N, 3)
        R_w2c = np.transpose(R_c2w, (0, 2, 1))  # (N, 3, 3)
        dP_w = P_w[1:] - P_w[:-1]          # (N-1, 3)
        dP_c = np.einsum("tij,tj->ti", R_w2c[:-1], dP_w)  # (N-1, 3)
        mags = np.linalg.norm(dP_c, axis=1)
        nonzero = mags > eps
        dbar = mags[nonzero].mean() if np.any(nonzero) else 1.0
        if dbar > eps:
            aligned[:, :3, 3] /= dbar
    elif mode == "none":
        pass
    else:
        raise ValueError(f"unknown camera normalization mode: {mode!r}")

    if mode != "none" and post_scale != 1.0:
        aligned[:, :3, 3] *= post_scale

    return aligned.astype(np.float32)


def _read_intrinsic_npz(npz_path):
    """Read an intrinsics npz file. Supported layouts:
       - {'K': [3,3]}                                : a 3x3 matrix
       - {'data': [1,4] or [N,4]} = [fx, fy, cx, cy] : converted to a 3x3 K
       - {'data': [3,3] or [N,3,3]}                  : a K matrix stored under 'data'
       Returns numpy [3, 3] intrinsics.
    """
    arr = np.load(npz_path)
    if "K" in arr:
        return arr["K"]
    if "data" in arr:
        d = arr["data"]
        if d.ndim == 3 and d.shape[-2:] == (3, 3):
            return d[0]                       # [N, 3, 3] → [3, 3]
        if d.ndim == 2 and d.shape == (3, 3):
            return d                          # [3, 3]
        if d.ndim == 2 and d.shape[-1] == 4:
            # [N, 4] = [fx, fy, cx, cy] → 3x3 K
            fx, fy, cx, cy = d[0]
            return np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0,  0,  1]], dtype=np.float32)
    raise KeyError(f"cannot parse intrinsics from {npz_path}: keys={list(arr.files)}")


def find_external_intrinsic(pose_path):
    """Look for an external intrinsics file next to or above the pose file (intrinsics.npz or intrinsics/).

    Used by datasets whose pose npz carries no intrinsics; they are stored either as:
      - intrinsics.npz in the parent directory, or
      - intrinsics/{name}.npz in the parent directory.
    """
    pose_dir = os.path.dirname(pose_path)
    basename = os.path.basename(pose_path)
    for candidate in [
        os.path.join(pose_dir, "intrinsics.npz"),
        os.path.join(os.path.dirname(pose_dir), "intrinsics.npz"),
    ]:
        if os.path.exists(candidate):
            return _read_intrinsic_npz(candidate)
    for candidate in [
        os.path.join(os.path.dirname(pose_dir), "intrinsics", basename),
        os.path.join(pose_dir, "intrinsics", basename),
        os.path.join(os.path.dirname(pose_dir), "pose_intrinsics", basename),
        os.path.join(pose_dir, "pose_intrinsics", basename),
    ]:
        if os.path.exists(candidate):
            return _read_intrinsic_npz(candidate)
    return None


class MultiSourceVideoDataset(Dataset):
    """
    Multi-source video dataset supporting sources with and without camera parameters.

    Source types (all carrying camera parameters):
        OpenVid, PusaV1, hailuo_mem_2k, veo3, mugen, spatialvid,
        sekai_real_walking, mugen_v2, sekai_game_walking, RealEstate10K, mp4_frame_game_3

    Args:
        video_base_dir: video root directory (data/Video)
        annotation_base_dir: annotation root directory (data/Annotation)
        sources: which sources to load
        width, height: output resolution
        target_fps: target frame rate
        max_frames: maximum number of frames (None = read all)
        random_frames: sample the window start randomly

    Returns:
        - pixel_values: video frames [F, C, H, W]
        - ref_img: reference image [C, H, W]
        - caption: text prompt
        - intrinsic: normalized intrinsics [3, 3] or zeros
        - cam_c2w: normalized extrinsics [N, 4, 4] or zeros
        - videoid: video id
        - has_camera: whether camera parameters are present
        - source: source name
    """


    SOURCE_CONFIGS = {
        'sekai_real_hq': {
            'has_camera': True,
            'annotation_subdir': 'sekai_real_hq',
            'jsonl': 'sekai_real_hq.jsonl',
            'video_subdir': '',
            'caption_subdir': '',
            'pose_subdir': '',
            'original_width': 1280.0,
            'original_height': 720.0,
            'use_segment_caption': True,
            'segment_caption_field': 'full_prompt',
            'overall_caption_field': 'short_prompt',
        },


    }

    def __init__(
        self,
        video_base_dir="data/Video",
        annotation_base_dir="data/Annotation",
        sources=None,  # None = load every registered source
        width=832,
        height=480,
        target_fps=16,
        min_frames=None,  # minimum number of frames
        max_frames=None,  # None = read all frames
        allow_short_samples=False,
        vae_grid_align=False,
        random_frames=False,  # sample the window start randomly
        pose_extra_frames=0,
        prefer_max_frames=False,  # validation: always take the longest legal window
        original_width=1280.0,
        original_height=720.0,
        use_cache=True,  # cache the scanned sample list
        #   export ALAYA_DATASET_CACHE_DIR=/dev/shm/dataset_cache
        cache_dir=os.environ.get("ALAYA_DATASET_CACHE_DIR", ".cache/dataset"),
        skip_file_check=False,  # skip existence checks to speed up scanning
        abstract_caption_prob=0.0,  # probability of using the abstract_caption field
        return_raw_pose=False,  # validation: also return the un-normalized poses
        camera_norm_mode="max",  # extrinsics normalization mode: "max" / "relic" / "none"
        camera_post_relic_scale=1.0,  # global scale applied after relic normalization
        require_camera=False,  # require camera data (samples without it are skipped)
        vae_temporal_factor=8,  # VAE temporal compression factor
        cp_size=1,  # Context Parallel size, latent_frames % cp_size == 0
        mp4_frame_game_video_count=None,  # cap on videos taken from each trajectory folder
        camera_guided_sampling=False,  # prefer sub-windows containing turns or look-backs
        output_latent_frames=None,     # K (latent frames) used to read valid_k{K}_starts
        use_valid_starts=False,        # sample window starts from valid_k{K}_starts when present
        valid_starts_anchor_offset=0,  # legacy: lock the gap; only used when roll_layout=False
        roll_layout=False,             # pre-roll gap_steps and cond_mode so the output aligns exactly
        max_gap_latents=0,             # gap_steps roll range [0, max_gap_latents] when roll_layout
        min_gap_latents_hc=0,          # lower bound of gap_steps for history-conditioned samples
        min_gap_latents_i2v=0,         # lower bound of gap_steps for i2v samples
        i2v_prob=0.0,                  # probability used when rolling cond_mode
        sink_remote=False,             # replace the first sink frames with a distant frame of the video
        sink_remote_min_distance=8,    # minimum latent distance between the sink frame and the window
        sink_latent_frames=0,          # number of sink latents (drives sink_pixel_count)
        caption_anchor_frame=None,     # validation: use the segment containing this frame for the whole rollout
        event_target_anchor_frame=None, # event window: place the sampled target start at this local frame
    ):
        super().__init__()
        self.video_base_dir = video_base_dir
        self.annotation_base_dir = annotation_base_dir
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.allow_short_samples = allow_short_samples
        self.vae_grid_align = bool(vae_grid_align)
        self.random_frames = random_frames
        self.pose_extra_frames = int(pose_extra_frames)
        self.prefer_max_frames = bool(prefer_max_frames)
        self.original_width = original_width
        self.original_height = original_height
        self.skip_file_check = skip_file_check
        self.abstract_caption_prob = abstract_caption_prob
        self.return_raw_pose = return_raw_pose
        self.camera_norm_mode = camera_norm_mode
        self.camera_post_relic_scale = camera_post_relic_scale
        self.require_camera = require_camera
        self.camera_guided_sampling = camera_guided_sampling
        self.vae_temporal_factor = vae_temporal_factor
        self.cp_size = cp_size
        self.mp4_frame_game_video_count = mp4_frame_game_video_count
        self.output_latent_frames = output_latent_frames
        self.use_valid_starts = use_valid_starts
        self.valid_starts_anchor_offset = int(valid_starts_anchor_offset)
        self.roll_layout = bool(roll_layout)
        self.max_gap_latents = int(max_gap_latents)
        self.min_gap_latents_hc = int(min_gap_latents_hc)
        self.min_gap_latents_i2v = int(min_gap_latents_i2v)
        self.i2v_prob = float(i2v_prob)
        self.sink_remote = bool(sink_remote)
        self.sink_remote_min_distance = int(sink_remote_min_distance)
        self.sink_latent_frames_for_remote = int(sink_latent_frames)
        self.caption_anchor_frame = None if caption_anchor_frame is None else int(caption_anchor_frame)
        self.event_target_anchor_frame = (
            None if event_target_anchor_frame is None else int(event_target_anchor_frame)
        )
        self._seed_epoch = 0
        self._seed_epoch_shared = torch.zeros((), dtype=torch.long).share_memory_()

        self._to_tensor = transforms.ToTensor()

        # video_path → estimated target-fps frame count from jsonl metadata
        self._sample_target_frame_counts = {}

        self._valid_starts_map = {}

        if sources is None:
            sources = list(self.SOURCE_CONFIGS.keys())

        self.sources = sources
        self.samples = []  # [(video_path, caption_path, pose_path, source_name, video_id)]

        cache_loaded = False
        if use_cache:
            cache_loaded = self._load_from_cache(cache_dir, sources)
            if cache_loaded and self.samples:
                has_camera_sources = {
                    s for s in sources
                    if self.SOURCE_CONFIGS.get(s, {}).get('has_camera', False)
                    and not self.SOURCE_CONFIGS.get(s, {}).get('static_camera', False)
                }
                if has_camera_sources:
                    has_pose = any(s[2] is not None for s in self.samples if s[3] in has_camera_sources)
                    if not has_pose:
                        print(f"[MultiSourceVideoDataset] every cached pose_path is None; discarding the cache")
                        self.samples = []
                        cache_loaded = False

        if not cache_loaded:
            _scan_progress = 0   # progress counter for the initial scan
            _scan_t0 = time.time()
            import os as _os_rank
            _scan_rank0 = _os_rank.environ.get('RANK', '0') == '0'
            for source in sources:
                if source not in self.SOURCE_CONFIGS:
                    print(f"[MultiSourceVideoDataset] Warning: Unknown source '{source}', skipping")
                    continue
                self._load_source(source)

            if use_cache:
                self._save_to_cache(cache_dir, sources)

        _shuffle_rng = random.Random(42)
        _shuffle_rng.shuffle(self.samples)
        print(f"[MultiSourceVideoDataset] Loaded {len(self.samples)} samples from {sources}")

    def _get_cache_path(self, cache_dir, sources):
        """Build the cache file path."""
        sources_key = "_".join(sorted(sources))
        source_keys = []
        for source in sorted(sources):
            config = self.SOURCE_CONFIGS.get(source, {})
            source_keys.extend([
                source,
                (hashlib.md5("|".join(config.get("jsonl")).encode()).hexdigest()[:12]
                 if isinstance(config.get("jsonl"), list) else str(config.get("jsonl", ""))),
                str(config.get("pose_subdir", "")),
            ])
        if source_keys:
            sources_key = "_".join(part.replace("/", "_") for part in source_keys)
        return os.path.join(cache_dir, f"multi_source_{sources_key}.pkl")

    def _load_from_cache(self, cache_dir, sources):
        """Load the sample list from cache."""
        cache_path = self._get_cache_path(cache_dir, sources)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.samples = cached_data['samples']
                    self._sample_target_frame_counts = cached_data.get('sample_target_frame_counts', {}) or {}
                    self._valid_starts_map = cached_data.get('valid_starts_map', {}) or {}
                    print(f"[MultiSourceVideoDataset] Loaded {len(self.samples)} samples from cache: {cache_path} "
                          f"frame_count_meta={len(self._sample_target_frame_counts)} "
                          f"valid_starts_map={len(self._valid_starts_map)})")
                    return True
            except Exception as e:
                print(f"[MultiSourceVideoDataset] Failed to load cache: {e}")
        return False

    def _save_to_cache(self, cache_dir, sources):
        """Save the sample list to cache."""
        try:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = self._get_cache_path(cache_dir, sources)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'samples': self.samples,
                    'sample_target_frame_counts': self._sample_target_frame_counts,
                    'valid_starts_map': self._valid_starts_map,
                }, f)
            print(f"[MultiSourceVideoDataset] Saved {len(self.samples)} samples to cache: {cache_path} "
                  f"frame_count_meta={len(self._sample_target_frame_counts)} "
                  f"valid_starts_map={len(self._valid_starts_map)})")
        except Exception as e:
            print(f"[MultiSourceVideoDataset] Failed to save cache: {e}")

    def _load_source(self, source_name):
        """Load a single source."""
        config = self.SOURCE_CONFIGS[source_name]
        _ann_subdir = config.get('annotation_subdir', source_name)
        annotation_dir = os.path.join(self.annotation_base_dir, _ann_subdir)
        video_dir = os.path.join(self.video_base_dir, config['video_subdir'])
        caption_dir = os.path.join(annotation_dir, config['caption_subdir'])
        pose_dir = os.path.join(annotation_dir, config.get('pose_subdir', '')) if config['has_camera'] else None

        jsonl_config = config['jsonl']
        jsonl_list = jsonl_config if isinstance(jsonl_config, list) else [jsonl_config]
        _jsonl_abs = config.get('jsonl_absolute', False)
        for jsonl_name in jsonl_list:
            jsonl_path = jsonl_name if _jsonl_abs else os.path.join(annotation_dir, jsonl_name)
            if os.path.exists(jsonl_path):
                self._load_from_jsonl(source_name, jsonl_path, video_dir, caption_dir, pose_dir)
            else:
                print(f"[MultiSourceVideoDataset] Warning: {jsonl_path} not found")

    def _load_from_jsonl(self, source_name, jsonl_path, video_dir, caption_dir, pose_dir):
        """Load samples from a jsonl file.

        Path resolution (every path comes from the jsonl):
        - video:  'video'/'video_path' field; absolute paths are used as-is, relative ones join video_base_dir
        - prompt: 'prompt'/'prompt_path' field; absolute paths are used as-is, relative ones join annotation_base_dir
        - pose:   'pose'/'pose_path' field; if absent it is derived from pose_dir/{rel_subdir}/{video_name}.npz
        """
        config = self.SOURCE_CONFIGS[source_name]
        count = 0
        skipped = 0
        video_path_replace = config.get('video_path_replace', {})
        caption_ext = config.get('caption_ext', '.json')
        video_subdir = config.get('video_subdir', '')

        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())

                    # === Video path ===
                    video_rel = item.get('video', '') or item.get('video_path', '')
                    if not video_rel:
                        skipped += 1
                        continue

                    for old_prefix, new_prefix in video_path_replace.items():
                        if video_rel.startswith(old_prefix):
                            video_rel = new_prefix + video_rel[len(old_prefix):]
                            break

                    if os.path.isabs(video_rel):
                        full_video_path = video_rel
                    else:
                        full_video_path = os.path.join(self.video_base_dir, video_rel)

                    if not full_video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        skipped += 1
                        continue

                    if not self.skip_file_check and not os.path.exists(full_video_path):
                        skipped += 1
                        continue

                    video_name = os.path.splitext(os.path.basename(full_video_path))[0]

                    rel_subdir = ''
                    if video_subdir and not os.path.isabs(video_rel) and video_rel.startswith(video_subdir + '/'):
                        rel_from_subdir = video_rel[len(video_subdir) + 1:]
                        rel_subdir = os.path.dirname(rel_from_subdir)

                    prompt_field = item.get('prompt') or item.get('prompt_path')
                    prompt_path_replace = config.get('prompt_path_replace', {})
                    if prompt_field:
                        _looks_like_inline = (
                            not os.path.isabs(prompt_field)
                            and '/' not in prompt_field
                            and not prompt_field.lower().endswith(('.json', '.txt'))
                            and ' ' in prompt_field
                        )
                        if _looks_like_inline:
                            caption_path = f"__INLINE__:{prompt_field}"
                        else:
                            for old_p, new_p in prompt_path_replace.items():
                                if prompt_field.startswith(old_p):
                                    prompt_field = new_p + prompt_field[len(old_p):]
                                    break
                            if os.path.isabs(prompt_field):
                                caption_path = prompt_field
                            elif '/' in prompt_field:
                                caption_path = os.path.join(self.annotation_base_dir, prompt_field)
                            else:
                                caption_path = os.path.join(caption_dir, prompt_field)
                    else:
                        inline_caption = item.get('overall_caption') or item.get('caption') or item.get('text')
                        if inline_caption:
                            caption_path = f"__INLINE__:{inline_caption}"
                        elif rel_subdir:
                            caption_path = os.path.join(caption_dir, rel_subdir, f"{video_name}{caption_ext}")
                        else:
                            caption_path = os.path.join(caption_dir, f"{video_name}{caption_ext}")

                    pose_path = None
                    if pose_dir and not config.get('static_camera', False):
                        pose_field = item.get('pose') or item.get('pose_path')
                        pose_path_replace = config.get('pose_path_replace', {})
                        if pose_field:
                            for old_p, new_p in pose_path_replace.items():
                                if pose_field.startswith(old_p):
                                    pose_field = new_p + pose_field[len(old_p):]
                                    break
                            if os.path.isabs(pose_field):
                                pose_path = pose_field
                            else:
                                pose_path = os.path.join(self.annotation_base_dir, pose_field)
                        else:
                            if rel_subdir:
                                pose_path = os.path.join(pose_dir, rel_subdir, f"{video_name}.npz")
                            else:
                                pose_path = os.path.join(pose_dir, f"{video_name}.npz")

                    self.samples.append((full_video_path, caption_path, pose_path, source_name, video_name))
                    _scan_progress += 1
                    if _scan_rank0 and _scan_progress % 2000 == 0:
                        print(f"[MultiSourceVideoDataset] scanning... {_scan_progress} samples "
                              f"({time.time() - _scan_t0:.0f}s)", flush=True)
                    count += 1
                    target_frame_count = self._estimate_target_frame_count(item)
                    if target_frame_count is not None:
                        self._sample_target_frame_counts[full_video_path] = int(target_frame_count)

                    if self.use_valid_starts and self.output_latent_frames is not None:
                        _vstart_field = f"valid_k{int(self.output_latent_frames)}_starts"
                        _vstart_val = item.get(_vstart_field)
                        if _vstart_val:
                            self._valid_starts_map[full_video_path] = list(_vstart_val)




                except Exception as e:
                    continue

        if source_name == 'mp4_frame_game_3' and self.mp4_frame_game_video_count is not None:
            from collections import defaultdict
            folder_counts = defaultdict(list)
            other_samples = []
            for sample in self.samples:
                if sample[3] == 'mp4_frame_game_3':
                    folder_name = os.path.basename(os.path.dirname(sample[0]))
                    folder_counts[folder_name].append(sample)
                else:
                    other_samples.append(sample)

            limited_samples = []
            for folder_name, samples in sorted(folder_counts.items()):
                limited_samples.extend(samples[:self.mp4_frame_game_video_count])

            original_count = count
            count = len(limited_samples)
            self.samples = other_samples + limited_samples
            print(f"[MultiSourceVideoDataset] per-folder cap {self.mp4_frame_game_video_count}: "
                  f"{original_count} -> {count} samples ({len(folder_counts)} trajectory folders)")

        if skipped > 0:
            print(f"[MultiSourceVideoDataset] Loaded {count} samples from {source_name} (skipped {skipped} invalid entries)")
        else:
            print(f"[MultiSourceVideoDataset] Loaded {count} samples from {source_name}")

    def _estimate_target_frame_count(self, item):
        def _as_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        duration = _as_float(item.get('video_duration') or item.get('duration'))
        if duration is not None and duration > 0:
            return max(1, int(duration * float(self.target_fps)))

        frame_count = (
            item.get('num_frames')
            or item.get('frame_count')
            or item.get('video_frames')
            or item.get('frames')
        )
        frame_count = _as_float(frame_count)
        if frame_count is None or frame_count <= 0:
            return None
        fps = _as_float(item.get('video_fps') or item.get('fps') or item.get('source_fps'))
        if fps is not None and fps > 0 and float(self.target_fps) > 0:
            return max(1, int(frame_count * float(self.target_fps) / fps))
        return max(1, int(frame_count))

    def _process_frame(self, img):
        """Convert a PIL image to a tensor and resize it with bicubic interpolation."""
        tensor = self._to_tensor(img)
        tensor = F.interpolate(tensor.unsqueeze(0), size=(self.height, self.width),
                              mode='bicubic', align_corners=False, antialias=True)
        return tensor.squeeze(0)

    def _load_caption(self, caption_path):
        """Load a caption.

        Returns:
            (str, str): (caption_text, caption_type)
            caption_type: "abstract" | "overall"

        Raises:
            RuntimeError: raised when the caption file is missing or has no usable field; __getitem__ retries
        """
        if caption_path is None:
            raise RuntimeError("caption_path is None, skip this sample")

        if caption_path.startswith("__INLINE__:"):
            return caption_path[len("__INLINE__:"):], "overall"

        if not os.path.exists(caption_path):
            if 'caption_HQ' in caption_path:
                fallback_path = caption_path.replace('caption_HQ', 'caption')
                if os.path.exists(fallback_path):
                    caption_path = fallback_path
                else:
                    raise RuntimeError(f"caption file not found: {caption_path} (fallback also missing)")
            else:
                raise RuntimeError(f"caption file not found: {caption_path}")

        with open(caption_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if self.abstract_caption_prob > 0 and random.random() < self.abstract_caption_prob:
                abstract = data.get("abstract_caption")
                if abstract:
                    return abstract, "abstract"
            caption = data.get("overall_caption") or data.get("caption") or data.get("text")
            if not caption and isinstance(data.get("overall"), dict):
                caption = data["overall"].get("description")
            if caption:
                return caption, "overall"
            else:
                raise RuntimeError(f"no caption field in {caption_path}, keys: {list(data.keys())}")

    def _load_camera_params(self, pose_path, frame_ids, source_config=None, strict_ids=False):
        """Load camera parameters.

        Args:
            pose_path: path to the pose file
            frame_ids: frame indices to read
            source_config: optional source config, used for the per-source resolution

        Returns:
            When self.return_raw_pose=False: (intrinsic, cam_c2w, orig_w, orig_h)
            When self.return_raw_pose=True:  (intrinsic, cam_c2w, orig_w, orig_h, intrinsic_raw, cam_c2w_raw)
        """
        if pose_path is None:
            if self.require_camera:
                raise RuntimeError(f"pose_path is None while require_camera=True; skipping this sample")
            if self.return_raw_pose:
                return None, None, 0.0, 0.0, None, None
            return None, None, 0.0, 0.0
        if not os.path.exists(pose_path):
            if self.require_camera:
                raise RuntimeError(f"pose file not found: {pose_path}, require_camera=True; skipping this sample")
            print(f"[Camera Debug] pose file not found: {pose_path}", flush=True)
            if self.return_raw_pose:
                return None, None, 0.0, 0.0, None, None
            return None, None, 0.0, 0.0

        try:
            data = np.load(pose_path)

            if "intrinsics" in data:
                intrinsics = data["intrinsics"]
                intrinsic = intrinsics[min(frame_ids[0], len(intrinsics)-1)] if intrinsics.ndim == 3 else intrinsics
            elif "intrinsic" in data:
                intrinsic = data["intrinsic"]
            elif "K" in data:
                intrinsic = data["K"]
            else:
                intrinsic = find_external_intrinsic(pose_path)
                if intrinsic is None:
                    if source_config and source_config.get('fallback_default_intrinsic', False):
                        intrinsic = _default_normalized_intrinsic()
                    else:
                        raise RuntimeError("no intrinsics found: the pose file has no intrinsics/intrinsic/K key and no intrinsics.npz")

            if "cam_c2w" in data:
                cam_c2w = data["cam_c2w"]
            elif "extrinsic" in data:
                cam_c2w = data["extrinsic"]
            elif "data" in data:
                cam_c2w = data["data"]
            else:
                raise RuntimeError(f"no extrinsics found: the pose file has no cam_c2w/extrinsic/data key, keys={list(data.files)}")

            fx_val = float(intrinsic[0, 0])
            fy_val = float(intrinsic[1, 1])
            if fx_val <= 0 or fy_val <= 0 or np.isnan(fx_val) or np.isnan(fy_val):
                raise RuntimeError(f"invalid intrinsics: fx={fx_val}, fy={fy_val}; skipping this sample")

            if strict_ids and frame_ids and int(max(frame_ids)) >= len(cam_c2w):
                raise RuntimeError(
                    f"pose file too short: max_frame_id={int(max(frame_ids))} >= pose_frames={len(cam_c2w)}; skipping"
                )
            valid_ids = [min(i, len(cam_c2w)-1) for i in frame_ids]
            cam_c2w = cam_c2w[valid_ids]

            cx = float(intrinsic[0, 2])
            cy = float(intrinsic[1, 2])
            if cx > 1.0 and cy > 1.0:
                orig_w = cx * 2.0
                orig_h = cy * 2.0
            else:
                orig_w = source_config.get('original_width', self.original_width) if source_config else self.original_width
                orig_h = source_config.get('original_height', self.original_height) if source_config else self.original_height

            if self.return_raw_pose:
                intrinsic_raw = intrinsic.copy().astype(np.float32)
                cam_c2w_raw = cam_c2w.copy().astype(np.float32)

            cx_val = float(intrinsic[0, 2])
            if cx_val > 1.0:
                w_est = cx_val * 2.0
                h_est = float(intrinsic[1, 2]) * 2.0
                intrinsic[0, 0] /= w_est   # fx
                intrinsic[1, 1] /= h_est   # fy
                intrinsic[0, 2] = 0.5      # cx
                intrinsic[1, 2] = 0.5      # cy

            cam_c2w = normalize_cam_c2w(cam_c2w, mode=self.camera_norm_mode,
                                          post_scale=self.camera_post_relic_scale)

            if self.return_raw_pose:
                return intrinsic.astype(np.float32), cam_c2w, orig_w, orig_h, intrinsic_raw, cam_c2w_raw
            return intrinsic.astype(np.float32), cam_c2w, orig_w, orig_h
        except Exception as e:
            if self.require_camera:
                raise RuntimeError(f"Error loading pose {pose_path}: {e}, require_camera=True; skipping this sample")
            print(f"[MultiSourceVideoDataset] Error loading pose {pose_path}: {e}")
            if self.return_raw_pose:
                return None, None, 0.0, 0.0, None, None
            return None, None, 0.0, 0.0

    VIDEO_READ_TIMEOUT = 120

    @staticmethod
    def _alarm_handler(signum, frame):
        """SIGALRM handler: interrupt a hung C-library call (e.g. the video reader)."""
        raise TimeoutError("Video reading timed out (SIGALRM)")

    def get_sample(self, index):
        video_path, caption_path, pose_path, source_name, video_id = self.samples[index]
        config = self.SOURCE_CONFIGS[source_name]

        old_handler = signal.signal(signal.SIGALRM, self._alarm_handler)
        signal.alarm(self.VIDEO_READ_TIMEOUT)

        try:
            return self._get_sample_inner(video_path, caption_path, pose_path,
                                          source_name, video_id, config,
                                          sample_index=index)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _get_sample_inner(self, video_path, caption_path, pose_path,
                          source_name, video_id, config, sample_index=None):
        try:
            video_reader = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            raise RuntimeError(f"Error reading {video_path}: {e}")
        video_length = len(video_reader)
        video_fps = video_reader.get_avg_fps()


        _fps_ratio = video_fps / self.target_fps if self.target_fps > 0 else 1.0
        _need_interp = _fps_ratio < 1.0  # interpolation needed (e.g. 16fps -> 24fps)
        sample_rate = max(1, int(_fps_ratio))
        available_frames = int(video_length / max(_fps_ratio, 1.0)) if not _need_interp else int(video_length * self.target_fps / video_fps)

        # === Segment-level caption + frame-range sampling (e.g. wm_game_v1) ===
        seg_cached_caption = None
        seg_caption_type = "segment"
        seg_frame_low, seg_frame_high = None, None
        _overall_caption_text = None
        if config.get('use_segment_caption', False) and caption_path \
                and not caption_path.startswith("__INLINE__:"):
            try:
                if os.path.exists(caption_path):
                    with open(caption_path, 'r', encoding='utf-8') as _sf:
                        _seg_data = json.load(_sf)
                    _seg_field = config.get('segment_caption_field', 'scene_description')
                    _ov_field = config.get('overall_caption_field', 'short_prompt')

                    def _env_float(name, default):
                        try:
                            return float(os.environ.get(name, str(default)))
                        except (ValueError, TypeError):
                            return float(default)

                    def _overall_caption():
                        _ov = _seg_data.get('overall')
                        if isinstance(_ov, dict):
                            return (
                                _ov.get(_ov_field)
                                or _ov.get('full_prompt')
                                or _ov.get('description')
                                or _ov.get('short_prompt')
                            )
                        return (
                            _seg_data.get('overall_caption')
                            or _seg_data.get('caption')
                            or _seg_data.get('text')
                        )

                    def _segment_caption(seg, field):
                        return (
                            seg.get(field)
                            or seg.get('full_prompt')
                            or seg.get('scene_description')
                            or seg.get('short_prompt')
                        )

                    def _timed_segments(segs):
                        return [
                            s for s in (segs or [])
                            if _segment_caption(s, _seg_field)
                            and s.get('time_range_s')
                            and len(s['time_range_s']) == 2
                        ]

                    if self.caption_anchor_frame is not None:
                        # Validation rollout: use the text for the first target chunk.
                        # This must not crop/shorten the loaded video; all rollout chunks
                        # reuse the same encoded context.
                        _anchor_sec = float(self.caption_anchor_frame) / max(float(self.target_fps), 1.0)
                        _chosen = None
                        _candidate_groups = [
                            _timed_segments(_seg_data.get('segments') or []),
                            _timed_segments(_seg_data.get('merged_segments') or []),
                        ]
                        for _group in _candidate_groups:
                            _inside = [
                                s for s in _group
                                if float(s['time_range_s'][0]) <= _anchor_sec < float(s['time_range_s'][1])
                            ]
                            if _inside:
                                _chosen = min(
                                    _inside,
                                    key=lambda s: float(s['time_range_s'][1]) - float(s['time_range_s'][0]),
                                )
                                break
                        if _chosen is None:
                            _all = [s for group in _candidate_groups for s in group]
                            if _all:
                                _chosen = min(
                                    _all,
                                    key=lambda s: min(
                                        abs(_anchor_sec - float(s['time_range_s'][0])),
                                        abs(_anchor_sec - float(s['time_range_s'][1])),
                                    ),
                                )
                        if _chosen is not None:
                            seg_cached_caption = _segment_caption(_chosen, _seg_field)
                            seg_caption_type = "segment_anchor"
                    elif random.random() < _env_float('LTX_OVERALL_CAPTION_PROB', 0.1):
                        _overall_caption_text = _overall_caption()
                    else:
                        _max_train_sec = (self.max_frames / max(self.target_fps, 1.0)) \
                            if self.max_frames else 1e9
                        _mseg = _seg_data.get('merged_segments') or []
                        _oseg = _seg_data.get('segments') or []
                        if _mseg and _oseg:
                            _merged_prob = _env_float('LTX_MERGED_SEGMENT_PROB', 0.75)
                            _segs = _mseg if random.random() < _merged_prob else _oseg
                        else:
                            _segs = _mseg or _oseg

                        _concat_prob = _env_float('LTX_SEGMENT_CONCAT_PROB', 0.0)
                        _concat_min_dur = _env_float('LTX_SEGMENT_CONCAT_MIN_DUR', 20.0)
                        _concat_field = os.environ.get('LTX_SEGMENT_CONCAT_FIELD', 'short_prompt')

                        _eligible = [
                            s for s in _segs
                            if _segment_caption(s, _concat_field)
                            and s.get('time_range_s')
                            and len(s['time_range_s']) == 2
                        ]
                        _sorted_segs = sorted(_eligible, key=lambda s: float(s['time_range_s'][0]))

                        _concat_chosen = None
                        if random.random() < _concat_prob and len(_sorted_segs) >= 2:
                            _valid_groups = []
                            for i in range(len(_sorted_segs)):
                                group = [_sorted_segs[i]]
                                for j in range(i + 1, len(_sorted_segs)):
                                    test_dur = (
                                        float(_sorted_segs[j]['time_range_s'][1])
                                        - float(group[0]['time_range_s'][0])
                                    )
                                    if test_dur > _max_train_sec:
                                        break
                                    group.append(_sorted_segs[j])
                                if len(group) >= 2:
                                    total_dur = (
                                        float(group[-1]['time_range_s'][1])
                                        - float(group[0]['time_range_s'][0])
                                    )
                                    if total_dur >= _concat_min_dur:
                                        _valid_groups.append(group)
                            if _valid_groups:
                                _concat_chosen = random.choice(_valid_groups)

                        if _concat_chosen is not None:
                            _t_lo = float(_concat_chosen[0]['time_range_s'][0])
                            _t_hi = float(_concat_chosen[-1]['time_range_s'][1])
                            _flo = max(0, int(_t_lo * video_fps))
                            _fhi = min(video_length, int(_t_hi * video_fps))
                            if _fhi > _flo:
                                seg_cached_caption = " ".join(
                                    str(_segment_caption(s, _concat_field))
                                    for s in _concat_chosen
                                )
                                seg_frame_low = _flo
                                seg_frame_high = _fhi
                        else:
                            _min_train_sec = (self.min_frames / max(self.target_fps, 1.0)) \
                                if self.min_frames else 0.0
                            _valid_segs = []
                            _fallback_segs = []
                            for s in _sorted_segs:
                                if not _segment_caption(s, _seg_field):
                                    continue
                                _dur = float(s['time_range_s'][1]) - float(s['time_range_s'][0])
                                if _dur <= 0:
                                    continue
                                _fallback_segs.append(s)
                                if _dur >= _min_train_sec:
                                    _valid_segs.append(s)
                            if not _valid_segs:
                                _valid_segs = _fallback_segs
                            if _valid_segs:
                                _chosen = random.choice(_valid_segs)
                                _t_lo, _t_hi = _chosen['time_range_s']
                                _flo = max(0, int(float(_t_lo) * video_fps))
                                _fhi = min(video_length, int(float(_t_hi) * video_fps))
                                if _fhi > _flo:
                                    _short_prob = _env_float('LTX_SEG_SHORT_PROMPT_PROB', 0.1)
                                    if (
                                        _short_prob > 0
                                        and random.random() < _short_prob
                                        and _chosen.get('short_prompt')
                                    ):
                                        seg_cached_caption = _chosen['short_prompt']
                                    else:
                                        seg_cached_caption = _segment_caption(_chosen, _seg_field)
                                    seg_frame_low = _flo
                                    seg_frame_high = _fhi
                            else:
                                _overall_caption_text = _overall_caption()

                        if seg_cached_caption is not None and seg_frame_low is not None and seg_frame_high is not None:
                            _seg_src_len = seg_frame_high - seg_frame_low
                            if _need_interp:
                                _seg_avail = int(_seg_src_len * self.target_fps / video_fps)
                            else:
                                _seg_avail = int(_seg_src_len / max(_fps_ratio, 1.0))
                            available_frames = min(available_frames, max(1, _seg_avail))
            except Exception:
                seg_cached_caption = None
                seg_frame_low = seg_frame_high = None
                _overall_caption_text = None

        vtf = self.vae_temporal_factor
        _cp = getattr(self, 'cp_size', 1)
        _step = vtf * _cp  # step between legal frame counts

        def to_compatible(f):
            """Round down to the largest frame count satisfying the VAE and context-parallel constraints."""
            # F = vtf * k + 1, (k+1) % cp_size == 0
            if _cp <= 1:
                adjusted = (f // vtf) * vtf + 1
                if adjusted > f:
                    adjusted -= vtf
            else:
                m = (f - 1 + vtf) // _step
                adjusted = _step * m - vtf + 1
                while adjusted > f and m > 0:
                    m -= 1
                    adjusted = _step * m - vtf + 1
            return max(vtf + 1, adjusted)  # at least vtf+1 frames

        if self.min_frames is not None and self.max_frames is not None:
            min_f = to_compatible(min(self.min_frames, available_frames))
            max_f = to_compatible(min(self.max_frames, available_frames))
            if self.prefer_max_frames:
                n_frames = max_f
            elif seg_cached_caption is not None:
                n_frames = max_f
            elif min_f < max_f:
                m_min = ((min_f - 1) // vtf + 1 + _cp - 1) // _cp
                m_max = ((max_f - 1) // vtf + 1) // _cp
                if m_max >= m_min:
                    m = random.randint(m_min, m_max)
                    n_frames = _step * m - vtf + 1
                else:
                    n_frames = max_f
            else:
                n_frames = max_f
        elif self.max_frames is not None:
            n_frames = to_compatible(min(self.max_frames, available_frames))
        else:
            n_frames = to_compatible(available_frames)

        n_frames = max(1, n_frames)

        _rolled_gap_steps = -1
        _rolled_cond_mode_id = -1

        _pose_extra_src = int(self.pose_extra_frames * _fps_ratio) + (1 if self.pose_extra_frames > 0 else 0)
        if self.pose_extra_frames > 0 and _need_interp:
            raise RuntimeError(
                f"pose_extra_frames>0 does not support interpolation from video_fps({video_fps}) < target_fps({self.target_fps}) "
                f"video={video_id}; remove this low-fps source from data.sources"
            )
        if _need_interp:
            n_source = int(n_frames * video_fps / self.target_fps) + 1
            _seg_lo = seg_frame_low if seg_frame_low is not None else 0
            _seg_hi = seg_frame_high if seg_frame_high is not None else video_length
            if seg_cached_caption is not None:
                if self.random_frames and n_source < (_seg_hi - _seg_lo):
                    max_start = _seg_hi - n_source
                    start = random.randint(_seg_lo, max_start) if max_start > _seg_lo else _seg_lo
                else:
                    start = _seg_lo
            elif self.random_frames and n_source < (_seg_hi - _seg_lo):
                max_start = _seg_hi - n_source - _pose_extra_src
                start = random.randint(_seg_lo, max_start) if max_start > _seg_lo else _seg_lo
            else:
                start = _seg_lo
            n_source = min(n_source, _seg_hi - start)
            frame_ids = list(range(start, start + n_source))
            _interp_src_start = start
            _interp_n_source = n_source
        else:
            def _uniform_sample(start_frame, n, ratio, total):
                ids = [min(int(start_frame + i * ratio), total - 1) for i in range(n)]
                return ids

            def _align_start_to_latent_grid(s, lo, hi, ratio):
                """Quantize the window start s down to the 8*ratio latent grid; out-of-range falls back to s."""
                if not self.vae_grid_align:
                    return s
                q = self.vae_temporal_factor * ratio
                if q <= 0 or abs(q - round(q)) > 1e-2:
                    return s  # non-integer grid (e.g. 25fps source): give up alignment and encode fresh
                q = int(round(q))
                aligned = (s // q) * q
                if aligned < lo:
                    aligned += q
                return aligned if lo <= aligned <= hi else s

            if seg_cached_caption is not None:
                _seg_lo = seg_frame_low if seg_frame_low is not None else 0
                _seg_hi = seg_frame_high if seg_frame_high is not None else video_length
                _window = int(n_frames * _fps_ratio) + _pose_extra_src
                if self.random_frames and _window < (_seg_hi - _seg_lo):
                    max_start = _seg_hi - _window
                    start = random.randint(_seg_lo, max_start) if max_start > _seg_lo else _seg_lo
                else:
                    start = _seg_lo
                self._last_camera_guided_score = -1.0
                start = _align_start_to_latent_grid(start, _seg_lo, max(_seg_lo, _seg_hi - _window), _fps_ratio)
                frame_ids = _uniform_sample(start, n_frames, _fps_ratio, video_length)
            elif self.random_frames and n_frames < available_frames:
                _window = int(n_frames * _fps_ratio) + _pose_extra_src
                _seg_lo = 0
                _seg_hi = video_length
                max_start = _seg_hi - _window
                _valid_starts_all = self._valid_starts_map.get(video_path) if self.use_valid_starts else None
                _use_cam_guided = (self.camera_guided_sampling and config['has_camera']
                                   and pose_path and max_start > _seg_lo)
                if self.roll_layout and _valid_starts_all and max_start > 0:
                    _rolled_cond_mode_id = 1 if random.random() < self.i2v_prob else 0
                    _cond_end_for_roll = 1 if _rolled_cond_mode_id == 1 else 0
                    _min_gap_for_roll = (
                        self.min_gap_latents_i2v
                        if _rolled_cond_mode_id == 1
                        else self.min_gap_latents_hc
                    )
                    _picked_ok = False
                    for _attempt in range(20):
                        _s_lat = random.choice(_valid_starts_all)
                        _max_gap_for_s = min(self.max_gap_latents, _s_lat - _cond_end_for_roll)
                        if _max_gap_for_s < _min_gap_for_roll:
                            continue
                        _gap_candidate = random.randint(_min_gap_for_roll, _max_gap_for_s)
                        _load_start_lat = _s_lat - _gap_candidate - _cond_end_for_roll
                        _cand_start = int(_load_start_lat * self.vae_temporal_factor * _fps_ratio)
                        if 0 <= _cand_start <= max_start:
                            _picked_ok = True
                            _rolled_gap_steps = _gap_candidate
                            start = _cand_start
                            break
                    if not _picked_ok:
                        start = random.randint(0, max_start) if max_start > 0 else 0
                        _rolled_gap_steps = -1
                        _rolled_cond_mode_id = -1
                    self._last_camera_guided_score = -1.0
                elif _valid_starts_all and max_start > 0:
                    _valid_starts = [s for s in _valid_starts_all if s >= self.valid_starts_anchor_offset]
                    if _valid_starts:
                        _s_lat = random.choice(_valid_starts)
                        start = int(
                            (_s_lat - self.valid_starts_anchor_offset)
                            * self.vae_temporal_factor * _fps_ratio
                        )
                        start = max(0, min(start, max_start))
                    else:
                        start = random.randint(0, max_start) if max_start > 0 else 0
                    self._last_camera_guided_score = -1.0
                elif _use_cam_guided:
                    try:
                        _raw_pose = np.load(pose_path)
                        _raw_c2w = None
                        for _pk in ('cam_c2w', 'extrinsic', 'data'):
                            if _pk in _raw_pose:
                                _raw_c2w = _raw_pose[_pk]
                                break
                        if _raw_c2w is not None and len(_raw_c2w) >= 2:
                            window_size = int(n_frames * _fps_ratio)
                            scored = _score_camera_subwindows(
                                _raw_c2w, n_frames, _fps_ratio, video_length,
                                window_size, smooth_kernel=7, angle_threshold_deg=15.0)
                            weights = [sc + 0.1 for _, sc in scored]  # +0.1 keeps every weight positive
                            chosen = random.choices(scored, weights=weights, k=1)[0]
                            start = min(chosen[0], max_start)
                            self._last_camera_guided_score = chosen[1]
                            self._last_camera_guided_start = start
                        else:
                            start = random.randint(0, max_start) if max_start > 0 else 0
                            self._last_camera_guided_score = -1.0
                    except Exception as _e:
                        start = random.randint(0, max_start) if max_start > 0 else 0
                        self._last_camera_guided_score = -1.0
                else:
                    start = random.randint(0, max_start) if max_start > 0 else 0
                    self._last_camera_guided_score = -1.0
                if _rolled_gap_steps == -1:  # the exact-alignment path leaves this untouched
                    start = _align_start_to_latent_grid(start, 0, max_start, _fps_ratio)
                frame_ids = _uniform_sample(start, n_frames, _fps_ratio, video_length)
            else:
                frame_ids = _uniform_sample(0, n_frames, _fps_ratio, video_length)
                self._last_camera_guided_score = -1.0

        if len(frame_ids) == 0:
            frame_ids = [0]

        pose_frame_ids = list(frame_ids)
        if self.pose_extra_frames > 0:
            if len(frame_ids) >= 2:
                _pose_step = (frame_ids[-1] - frame_ids[0]) / max(1, len(frame_ids) - 1)
            else:
                _pose_step = max(1.0, float(_fps_ratio))
            for _pi in range(len(frame_ids), len(frame_ids) + self.pose_extra_frames):
                _nid = int(frame_ids[0] + _pi * _pose_step)
                if _nid >= video_length:
                    raise RuntimeError(
                        f"pose horizon too short: need source frame {_nid} >= video_length {video_length} "
                        f"(pose_extra_frames={self.pose_extra_frames}), video={video_id}; skipping this sample"
                    )
                pose_frame_ids.append(_nid)

        frames = []
        try:
            batch = video_reader.get_batch(frame_ids).asnumpy()  # [N, H, W, C]
            frames = [Image.fromarray(batch[i]) for i in range(batch.shape[0])]
        except TimeoutError:
            raise  # a SIGALRM timeout must propagate
        except Exception:
            for fid in frame_ids:
                try:
                    frame = video_reader[fid].asnumpy()
                    frames.append(Image.fromarray(frame))
                except TimeoutError:
                    raise
                except:
                    if len(frames) > 0:
                        frames.append(frames[-1])

        if len(frames) == 0:
            raise RuntimeError(f"Failed to read any frame from {video_path}")

        pixel_values = torch.stack([self._process_frame(f) for f in frames], dim=0)

        if _need_interp and pixel_values.shape[0] != n_frames:
            pv_5d = pixel_values.unsqueeze(0).permute(0, 2, 1, 3, 4).float()  # [1, C, N_src, H, W]
            pv_interp = torch.nn.functional.interpolate(
                pv_5d, size=(n_frames, pixel_values.shape[2], pixel_values.shape[3]),
                mode='trilinear', align_corners=True
            )
            pixel_values = pv_interp.permute(0, 2, 1, 3, 4).squeeze(0).to(pixel_values.dtype)  # [n_frames, C, H, W]

        if (self.sink_remote and self.sink_latent_frames_for_remote > 0
                and not _need_interp):
            _sink_pix_count = self.sink_latent_frames_for_remote * self.vae_temporal_factor
            if pixel_values.shape[0] > _sink_pix_count:
                _main_lat_lo = int(start) // self.vae_temporal_factor
                _main_pix_end = int(start) + int(n_frames * _fps_ratio)
                _main_lat_hi = _main_pix_end // self.vae_temporal_factor + 1
                _excl_lo = max(0, _main_lat_lo - self.sink_remote_min_distance)
                _excl_hi = _main_lat_hi + self.sink_remote_min_distance
                _video_total_lat = max(0, video_length // self.vae_temporal_factor)
                _max_sink_start = _video_total_lat - self.sink_latent_frames_for_remote
                _candidates = []
                if _excl_lo - self.sink_latent_frames_for_remote >= 0:
                    _candidates.extend(range(0, _excl_lo - self.sink_latent_frames_for_remote + 1))
                if _excl_hi <= _max_sink_start:
                    _candidates.extend(range(_excl_hi, _max_sink_start + 1))
                if _candidates:
                    _sink_lat_start = random.choice(_candidates)
                    _sink_pix_start = _sink_lat_start * self.vae_temporal_factor
                    _sink_frame_ids = list(range(_sink_pix_start, _sink_pix_start + _sink_pix_count))
                    try:
                        _sb = video_reader.get_batch(_sink_frame_ids).asnumpy()
                        _sink_frames = [Image.fromarray(_sb[i]) for i in range(_sb.shape[0])]
                        _sink_pv = torch.stack(
                            [self._process_frame(f) for f in _sink_frames], dim=0
                        ).to(pixel_values.dtype)
                        pixel_values[:_sink_pix_count] = _sink_pv
                    except TimeoutError:
                        raise
                    except Exception:
                        pass  # on read failure keep the original (neighbouring) sink frame

        if seg_cached_caption is not None:
            caption, caption_type = seg_cached_caption, seg_caption_type
        elif _overall_caption_text is not None:
            caption, caption_type = _overall_caption_text, "overall"
        else:
            caption, caption_type = self._load_caption(caption_path)

        if isinstance(caption, str) and '<camera' in caption.lower():
            try:
                _cam_drop_prob = float(os.environ.get('LTX_CAMERA_DROP_CONTENT_PROB', '1.0'))
            except (ValueError, TypeError):
                _cam_drop_prob = 1.0
            if _cam_drop_prob > 0 and random.random() < _cam_drop_prob:
                caption = re.sub(r'<camera\b[^>]*>.*?</camera>', ' ', caption, flags=re.IGNORECASE | re.DOTALL)
            else:
                caption = re.sub(r'<camera\b[^>]*>(.*?)</camera>', r'\1', caption, flags=re.IGNORECASE | re.DOTALL)
            caption = re.sub(r'\s+', ' ', caption).strip()



        segment_prompts = None
        _caption_key = config.get('caption_key', None)
        if (_caption_key and _caption_key.startswith('interleave_caption')
                and caption_path
                and not caption_path.startswith("__INLINE__:")
                and seg_cached_caption is None):  # do not clash with the segment-caption mode
            try:
                if os.path.exists(caption_path):
                    with open(caption_path, 'r', encoding='utf-8') as _ipf:
                        _ip_data = json.load(_ipf)
                    _ip = _ip_data.get(_caption_key)
                    if isinstance(_ip, dict):
                        _crop_offset_target = int(round(frame_ids[0] / max(_fps_ratio, 1e-6)))
                        _n_frames_out = pixel_values.shape[0]
                        _segs_aligned = []
                        for _seg_key in sorted(_ip.keys(),
                                key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0):
                            _seg = _ip[_seg_key]
                            if not isinstance(_seg, dict): continue
                            if 'start_time' not in _seg or 'end_time' not in _seg: continue
                            _prompt_text = _seg.get('caption') or _seg.get('prompt') or ''
                            if not _prompt_text: continue
                            _seg_s_full = int(_seg['start_time'] * self.target_fps)
                            _seg_e_full = int(_seg['end_time']   * self.target_fps)
                            _s = max(0, _seg_s_full - _crop_offset_target)
                            _e = min(_n_frames_out, _seg_e_full - _crop_offset_target)
                            if _e > _s:
                                _segs_aligned.append({
                                    'start_frame': int(_s),
                                    'end_frame':   int(_e),
                                    'prompt':      str(_prompt_text),
                                })
                        if _segs_aligned:
                            segment_prompts = _segs_aligned
            except Exception:
                segment_prompts = None

        intrinsic, cam_c2w = None, None
        intrinsic_raw, cam_c2w_raw = None, None
        pose_orig_w, pose_orig_h = 0.0, 0.0
        if config.get('static_camera', False) and config['has_camera']:
            intrinsic = np.array(
                [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            cam_c2w = np.repeat(np.eye(4, dtype=np.float32)[None], len(pose_frame_ids), axis=0)
            pose_orig_w = float(config.get('original_width', self.original_width))
            pose_orig_h = float(config.get('original_height', self.original_height))
            if self.return_raw_pose:
                intrinsic_raw = intrinsic.copy()
                cam_c2w_raw = cam_c2w.copy()
        elif config['has_camera'] and pose_path:
            if self.return_raw_pose:
                intrinsic, cam_c2w, pose_orig_w, pose_orig_h, intrinsic_raw, cam_c2w_raw = self._load_camera_params(
                    pose_path, pose_frame_ids, source_config=config, strict_ids=self.pose_extra_frames > 0)
            else:
                intrinsic, cam_c2w, pose_orig_w, pose_orig_h = self._load_camera_params(
                    pose_path, pose_frame_ids, source_config=config, strict_ids=self.pose_extra_frames > 0)


        result = {
            'pixel_values': pixel_values,
            'ref_img': self._process_frame(frames[0]),
            'caption': caption,
            'caption_type': caption_type,
            'intrinsic': torch.from_numpy(intrinsic) if intrinsic is not None else torch.zeros(3, 3),
            'cam_c2w': torch.from_numpy(cam_c2w) if cam_c2w is not None else torch.zeros(len(pose_frame_ids), 4, 4),
            'videoid': video_id,
            'has_camera': config['has_camera'] and intrinsic is not None,
            'source': source_name,
            'pose_orig_w': float(pose_orig_w),
            'pose_orig_h': float(pose_orig_h),
            'frame_start': int(frame_ids[0]),
            'frame_end': int(frame_ids[-1]),
        }
        if self.return_raw_pose:
            result['intrinsic_raw'] = torch.from_numpy(intrinsic_raw) if intrinsic_raw is not None else torch.zeros(3, 3)
            result['cam_c2w_raw'] = torch.from_numpy(cam_c2w_raw) if cam_c2w_raw is not None else torch.zeros(len(pose_frame_ids), 4, 4)
        result['segment_prompts'] = segment_prompts
        result['rolled_gap_steps'] = int(_rolled_gap_steps)
        result['rolled_cond_mode_id'] = int(_rolled_cond_mode_id)
        return result

    def __getitem__(self, idx):
        _original_idx = int(idx)
        _seed_epoch = self._current_seed_epoch()
        _seed = (int(idx) * 7919 + _seed_epoch * 104729) % (2**31)
        random.seed(_seed)
        np.random.seed(_seed)

        _dbg_rank = os.environ.get('RANK', '?')

        max_retries = 50

        for retry in range(max_retries):
            video_path = self.samples[idx][0] if idx < len(self.samples) else "unknown"
            try:
                sample = self.get_sample(idx)

                pixel_values = sample['pixel_values'].sub_(0.5).div_(0.5)
                ref_img = pixel_values[0]

                if (
                    self.min_frames is not None
                    and not self.allow_short_samples
                    and int(pixel_values.shape[0]) < int(self.min_frames)
                ):
                    raise RuntimeError(
                        f"sample too short: frames={int(pixel_values.shape[0])} < min_frames={int(self.min_frames)}, "
                        f"video={sample.get('videoid', '?')}, source={sample.get('source', '?')}"
                    )

                _cp_sz = int(getattr(self, 'cp_size', 1))
                if _cp_sz > 1:
                    _pixel_T = int(pixel_values.shape[0])
                    _latent_T = (_pixel_T - 1) // 8 + 1
                    if _latent_T < _cp_sz:
                        raise RuntimeError(
                            f"latent_T={_latent_T} < cp_size={_cp_sz} (pixel_T={_pixel_T}), "
                            f"video={sample.get('videoid', '?')}, source={sample.get('source', '?')}"
                        )

                result = (
                    pixel_values,
                    ref_img,
                    sample['caption'],
                    sample['intrinsic'],
                    sample['cam_c2w'],
                    sample['videoid'],
                    sample['has_camera'],
                    sample['source'],
                    sample['caption_type'],
                    sample['pose_orig_w'],
                    sample['pose_orig_h'],
                    int(sample.get('frame_start', -1)),
                    int(sample.get('frame_end', -1)),
                )
                if self.return_raw_pose:
                    result = result + (
                        sample['intrinsic_raw'],
                        sample['cam_c2w_raw'],
                    )
                _seg_prompts = sample.get('segment_prompts')
                _seg_prompts_json = json.dumps(_seg_prompts) if _seg_prompts is not None else "null"
                result = result + (_seg_prompts_json,)
                result = result + (
                    int(sample.get('rolled_gap_steps', -1)),
                    int(sample.get('rolled_cond_mode_id', -1)),
                )
                for i, v in enumerate(result):
                    if v is None:
                        raise RuntimeError(f"__getitem__ result[{i}] is None, video={video_path}")

                if os.environ.get("ALAYA_DATASET_DEBUG", "0") == "1":
                    # CP-Debug: print sample summary for checking CP-group data consistency.
                    try:
                        print(f"[CP-DSDebug] rank={_dbg_rank} orig_idx={_original_idx} "
                              f"final_idx={idx} retry={retry} n_frames={pixel_values.shape[0]} "
                              f"video_id={sample.get('videoid', '?')} source={sample.get('source', '?')} "
                              f"fs={sample.get('frame_start','?')} fe={sample.get('frame_end','?')}",
                              flush=True)
                    except Exception:
                        pass

                return result
            except Exception as e:
                new_idx = (int(idx) * 13 + retry * 17 + 1) % len(self.samples)
                video_path = self.samples[idx][0] if idx < len(self.samples) else "unknown"
                print(f"[MultiSourceVideoDataset] __getitem__ retry {retry}/{max_retries}: "
                      f"rank={_dbg_rank} orig_idx={_original_idx} cur_idx={idx} -> new_idx={new_idx} | "
                      f"{type(e).__name__}: {e} | video={video_path}",
                      flush=True)
                idx = new_idx
                _seed_epoch = self._current_seed_epoch()
                _seed = (int(idx) * 7919 + _seed_epoch * 104729) % (2**31)
                random.seed(_seed)
                np.random.seed(_seed)

        print(f"[MultiSourceVideoDataset] __getitem__ failed after {max_retries} retries; returning skip", flush=True)
        skip = ("skip", "skip", "skip", "skip", "skip", "skip", "skip", "skip", "skip", "skip", "skip", -1, -1)
        if self.return_raw_pose:
            skip = skip + ("skip", "skip")
        return skip

    def set_epoch(self, epoch: int):
        self._seed_epoch = int(epoch)
        if hasattr(self, "_seed_epoch_shared"):
            self._seed_epoch_shared.fill_(int(epoch))

    def _current_seed_epoch(self) -> int:
        if hasattr(self, "_seed_epoch_shared"):
            return int(self._seed_epoch_shared.item())
        return int(getattr(self, "_seed_epoch", 0))

    def __len__(self):
        return len(self.samples)


# =============================================================================
# =============================================================================

class WeightedConcatDataset(Dataset):
    """Merge multiple datasets with target sampling probabilities via virtual oversampling.

    Each sub-dataset is assigned a contiguous range of virtual indices proportional
    to its target weight.  A DistributedSampler that shuffles these indices uniformly
    will therefore sample each sub-dataset at roughly the configured probability.

    Args:
        datasets_and_weights: list of (dataset, weight, name) tuples.
            *weight* values are automatically normalised so they sum to 1.
    """

    def __init__(self, datasets_and_weights):
        super().__init__()
        assert len(datasets_and_weights) > 0, "Need at least one dataset"

        # Filter out zero-weight or empty datasets to avoid division-by-zero.
        filtered = [(d, w, n) for d, w, n in datasets_and_weights if w > 0 and len(d) > 0]
        assert len(filtered) > 0, "All dataset weights are zero or all datasets are empty"

        datasets, weights, names = zip(*filtered)
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]

        # Choose a reference length so that the smallest sub-dataset is
        # neither excessively over-sampled nor under-represented.
        # Strategy: pick a total virtual size proportional to the largest
        # (real_size / target_weight) so every dataset is sampled at most
        # ~1x per virtual epoch.
        max_effective = max(
            len(d) / w for d, w in zip(datasets, norm_weights)
        )
        # Cap virtual size: DistributedSampler calls torch.randperm(total_virtual)
        # which allocates int64 memory.  A 10M cap keeps memory < 80 MB and
        # randperm time < 1s while still giving good sampling fidelity.
        MAX_VIRTUAL_SIZE = 10_000_000
        total_virtual = min(int(max_effective), MAX_VIRTUAL_SIZE)
        if int(max_effective) > MAX_VIRTUAL_SIZE:
            print(f"[WeightedConcatDataset] WARNING: capping virtual size from "
                  f"{int(max_effective):,} to {MAX_VIRTUAL_SIZE:,} to avoid OOM in sampler")

        self.datasets = list(datasets)
        self.names = list(names)
        self.norm_weights = norm_weights

        # Build segment table: cumulative start indices for each sub-dataset.
        self.segments = []  # (virtual_start, virtual_size, dataset_idx)
        offset = 0
        for i, (ds, w) in enumerate(zip(self.datasets, self.norm_weights)):
            vsize = max(1, int(round(total_virtual * w)))
            self.segments.append((offset, vsize, i))
            offset += vsize
        self._total_len = offset

    def __len__(self):
        return self._total_len

    def set_epoch(self, epoch: int):
        for dataset in self.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

    def __getitem__(self, idx):
        # Binary-ish search (few segments, linear is fine)
        for seg_start, seg_size, ds_idx in self.segments:
            if idx < seg_start + seg_size:
                local_idx = (idx - seg_start) % len(self.datasets[ds_idx])
                return self.datasets[ds_idx][local_idx]
        # Fallback (should never happen)
        ds_idx = len(self.datasets) - 1
        return self.datasets[ds_idx][idx % len(self.datasets[ds_idx])]

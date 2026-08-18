from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


_FP_CLAUSE = {
    "W": "The camera moves steadily forward through the scene.",
    "S": "The camera retreats backward.",
    "A": "The camera slides to the left, strafing sideways.",
    "D": "The camera slides to the right, strafing sideways.",
    "left": "The camera turns to the left, panning the view leftward.",
    "right": "The camera turns to the right, panning the view rightward.",
    "up": "The camera tilts upward.",
    "down": "The camera tilts downward.",
}
_TP_CLAUSE = {
    "W": "The {s} walks forward and the camera follows behind at the same speed, keeping it centred in frame.",
    "S": "The {s} backs toward the camera as the camera retreats, staying centred in frame.",
    "A": "The {s} and the camera slide together to the left, the {s} staying centred in frame.",
    "D": "The {s} and the camera slide together to the right, the {s} staying centred in frame.",
    "left": "The camera arcs around the {s} to the left, keeping it centred in frame.",
    "right": "The camera arcs around the {s} to the right, keeping it centred in frame.",
    "up": "The camera tilts upward while the {s} stays in frame.",
    "down": "The camera tilts downward while the {s} stays in frame.",
}
_TRANS_KEYS = ("W", "S", "A", "D")


def _short_subject(case: dict) -> str:
    desc = str(((case.get("settings", {}) or {}).get("subject", {}) or {}).get("desc", "")).strip()
    for pre in ("The main subject is ", "the main subject is "):
        if desc.startswith(pre):
            desc = desc[len(pre):]
    desc = desc.lstrip("aA ").strip()
    words = desc.split()
    return " ".join(words[:5]).rstrip(",.;") or "subject"


def _action_clause(action: str, *, third_person: bool, subject: str) -> str:
    table = _TP_CLAUSE if third_person else _FP_CLAUSE
    parts = [p.strip() for p in str(action).replace(",", "+").split("+") if p.strip()]
    norm = []
    for p in parts:
        if p in {"w", "a", "s", "d"}:
            p = p.upper()
        if p in table:
            norm.append(p)
    norm.sort(key=lambda p: 0 if p in _TRANS_KEYS else 1)
    clauses = [table[p].format(s=subject) for p in norm]
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    first = clauses[0].rstrip(".")
    rest = [c[0].lower() + c[1:].rstrip(".") for c in clauses[1:]]
    return first + " while " + " while ".join(rest) + "."


def _strip_camera(text: str) -> str:
    if not isinstance(text, str) or "<camera" not in text.lower():
        return text
    text = re.sub(r"<camera\b[^>]*>.*?</camera>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


class WBenchNaviDataset(Dataset):
    """WBench navigation split for generation-only validation.

    Three mutually exclusive sources for the first frame:
      1) native: use settings.initial_image from cases/*.json; each case carries its own poses
      2) image_dir: use every image in a directory (one sample each); poses are borrowed from pose_case_id
      3) sekai_jsonl + sekai_random_n: draw N videos at random and take their first frames;
         poses are borrowed from pose_case_id.
    In modes 2 and 3 every sample shares the actions and caption of pose_case_id.
    The source name stays wbench_navi so the trainer path is unchanged.
    """

    def __init__(
        self,
        *,
        root: str,
        width: int,
        height: int,
        frames: int,
        case_ids: list[str] | None = None,
        image_dir: str | None = None,
        pose_case_id: str | None = None,
        pose_actions: list[str] | None = None,
        sekai_jsonl: str | None = None,
        sekai_video_base: str | None = None,
        sekai_caption_base: str | None = None,
        sekai_random_n: int = 0,
        sekai_seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.width = int(width)
        self.height = int(height)
        self.frames = int(frames)
        self._to_tensor = transforms.ToTensor()
        self._rewritten = _load_rewritten_prompts(self.root)

        cases_dir = self.root / "cases"
        if not cases_dir.is_dir():
            raise FileNotFoundError(f"WBench cases dir not found: {cases_dir}")

        custom_first_frame = bool(image_dir) or int(sekai_random_n) > 0
        self.entries: list[dict[str, Any]] = []

        if custom_first_frame:
            if not pose_case_id:
                raise ValueError(
                    "custom first-frame mode (image_dir / sekai_random_n) requires dataset.pose_case_id"
                )
            borrow_case = self._load_case(str(pose_case_id))
            if pose_actions:
                borrowed_actions = [str(a) for a in pose_actions]
            else:
                borrowed_actions = _navigation_actions(borrow_case)
            if not borrowed_actions:
                raise ValueError(f"pose_case_id={pose_case_id} has no navigation actions")
            borrowed_caption = _build_caption(borrow_case)
            borrowed_perspective = str(
                (borrow_case.get("settings", {}) or {}).get(
                    "perspective", borrow_case.get("perspective", "third_person")
                )
            )
            common = {
                "actions": borrowed_actions,
                "perspective": borrowed_perspective,
                "pose_case_id": str(pose_case_id),
            }

            if image_dir:
                d = Path(image_dir)
                if not d.is_dir():
                    raise FileNotFoundError(f"image_dir not found: {d}")
                imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
                if not imgs:
                    raise ValueError(f"no images found under image_dir={d}")
                caps: dict[str, str] = {}
                capfile = d / "captions.json"
                if capfile.exists():
                    try:
                        caps = json.load(open(capfile, "r", encoding="utf-8"))
                    except Exception:
                        caps = {}
                capdir = d / "captions"
                if capdir.is_dir():
                    for txt_path in sorted(capdir.glob("*.txt")):
                        text = _read_caption_txt(txt_path)
                        if text:
                            caps.setdefault(txt_path.stem, text)
                for p in imgs:
                    cap = caps.get(p.name) or caps.get(p.stem) or borrowed_caption
                    self.entries.append(
                        {"kind": "image", "path": str(p), "disp_id": f"img_{p.stem}", "caption": cap, **common}
                    )
            else:
                if not sekai_jsonl:
                    raise ValueError("sekai_random_n>0 requires dataset.sekai_jsonl")
                recs: list[dict[str, Any]] = []
                with open(sekai_jsonl, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            recs.append(json.loads(line))
                if not recs:
                    raise ValueError(f"empty sekai jsonl: {sekai_jsonl}")
                random.Random(int(sekai_seed)).shuffle(recs)
                picked = recs[: int(sekai_random_n)]
                vbase = Path(sekai_video_base) if sekai_video_base else None
                cbase = Path(sekai_caption_base) if sekai_caption_base else None
                for rec in picked:
                    vp = rec.get("video_path")
                    if not vp:
                        continue
                    vpath = Path(vp)
                    if not vpath.is_absolute() and vbase is not None:
                        vpath = vbase / vp
                    cap = borrowed_caption
                    pp = rec.get("prompt_path")
                    if pp and cbase is not None:
                        cap = _read_sekai_description(cbase / pp) or borrowed_caption
                    self.entries.append(
                        {"kind": "video", "path": str(vpath), "disp_id": f"sekai_{Path(vp).stem}", "caption": cap, **common}
                    )
                if not self.entries:
                    raise ValueError("no usable sekai videos picked")
            self.turn_counts = [len(e["actions"]) for e in self.entries]
            self.full_turn_counts = list(self.turn_counts)
        else:
            wanted = {str(x) for x in case_ids} if case_ids else None
            for path in sorted(cases_dir.glob("case_*.json"), key=_case_sort_key):
                case_id = path.stem.replace("case_", "")
                if wanted is not None and case_id not in wanted:
                    continue
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if _is_navigation_case(data):
                    self.entries.append({"kind": "case", "case_id": case_id, "case_path": str(path), "case": data})
            if not self.entries:
                raise ValueError(f"no WBench navigation cases found under {cases_dir}")
            self.turn_counts = [_navigation_turn_count(e["case"]) for e in self.entries]
            self.full_turn_counts = [max(1, _max_turn(e["case"])) for e in self.entries]

        self.cases = [
            (e["case_id"], e["case_path"], e["case"]) for e in self.entries if e.get("kind") == "case"
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def _load_case(self, case_id: str) -> dict[str, Any]:
        p = self.root / "cases" / f"case_{case_id}.json"
        if not p.exists():
            raise FileNotFoundError(f"pose_case_id case not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _read_first_frame(self, video_path: str) -> Image.Image:
        import decord  # imported lazily

        vr = decord.VideoReader(str(video_path))
        frame = vr[0].asnumpy()  # [H, W, 3] uint8 RGB
        return Image.fromarray(frame).convert("RGB")

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[int(index)]
        if entry.get("kind") == "case":
            return self._getitem_case(entry)
        return self._getitem_custom(entry)

    def _getitem_case(self, entry: dict[str, Any]) -> dict[str, Any]:
        case_id, case_path, case = entry["case_id"], entry["case_path"], entry["case"]
        settings = case.get("settings", {}) or {}
        image_rel = settings.get("initial_image")
        if not image_rel:
            raise RuntimeError(f"WBench case {case_id} has no settings.initial_image")
        image_path = Path(image_rel)
        if not image_path.is_absolute():
            image_path = self.root / image_path
        if not image_path.exists():
            fallback = self.root / "images" / f"case_{case_id}.jpg"
            if fallback.exists():
                image_path = fallback
        image = Image.open(image_path).convert("RGB")
        actions = _navigation_actions(case)
        rw = self._rewritten.get(str(case_id))  # offline training-style rewrite for this case, if any
        caption = _build_training_style_caption(case, rewritten=rw)  # training-style base prompt
        prompt_schedule = _build_interaction_prompt_schedule(case, rewritten=rw)  # per-turn interaction schedule
        subject_mask_path = ""
        subject_mask_rel = settings.get("subject_mask")
        if subject_mask_rel:
            candidate = Path(subject_mask_rel)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            if not candidate.exists():
                fallback = self.root / "masks" / f"case_{case_id}_mask.png"
                if fallback.exists():
                    candidate = fallback
            if candidate.exists():
                subject_mask_path = str(candidate)
        import os as _os
        if _os.environ.get("WBENCH_ACTION_CLAUSES", "0") == "1":
            persp0 = str((case.get("settings", {}) or {}).get("perspective", case.get("perspective", "third_person")))
            subj = _short_subject(case)
            nav_turns = {int(it.get("turn", 0) or 0) for it in (case.get("interactions") or []) if str(it.get("type")) == "navigation"}
            full_acts = _full_turn_actions(case)
            for ti in range(len(prompt_schedule)):
                if (ti + 1) in nav_turns and ti < len(full_acts):
                    cl = _action_clause(full_acts[ti], third_person=persp0 != "first_person", subject=subj)
                    if cl:
                        prompt_schedule[ti] = (prompt_schedule[ti] + " " + cl).strip()
        if _os.environ.get("WBENCH_OFFSCREEN_EVENTS", "0") == "1":
            _off = str(((case.get("scene_adherence") or {}).get("offscreen_part") or "")).strip()
            _pj = _os.environ.get("WBENCH_OFFSCREEN_PRESENCE_JSONL", "")
            _front = False
            if _pj:
                try:
                    import json as _json
                    for _line in open(_pj):
                        _d = _json.loads(_line)
                        if str(_d.get("case_id")) == str(case_id):
                            _off = str(_d.get("text") or "").strip() or _off
                            _front = True
                            break
                except Exception as _e:
                    print(f"[offscreen_presence] skip (fail-open): {_e}", flush=True)
            if _off and prompt_schedule:
                if _front:
                    prompt_schedule[0] = (_off + " " + prompt_schedule[0]).strip()
                else:
                    prompt_schedule[0] = (prompt_schedule[0] + " " + _off).strip()
        _tj = _os.environ.get("WBENCH_TURN0_PROMPT_JSONL", "")
        if _tj and prompt_schedule:
            try:
                import json as _json2
                for _line in open(_tj):
                    _d = _json2.loads(_line)
                    if str(_d.get("case_id")) == str(case_id):
                        _t = str(_d.get("text") or "").strip()
                        if _t:
                            prompt_schedule[0] = _t
                        break
            except Exception as _e:
                print(f"[turn0_prompt_override] skip (fail-open): {_e}", flush=True)
        if _os.environ.get("WBENCH_PHYSICS_CLAUSES", "0") == "1":
            _PHYS = {
                "surface_interaction": "Every footstep presses a crisp, lasting imprint into the ground surface, and each contact with water sends small spreading ripples.",
                "deformation": "Soft objects visibly deform under contact and slowly recover their original shape afterwards.",
            }
            _dims = ((case.get("causal_fidelity") or {}).get("dims") or [])
            _cl = " ".join(_PHYS[d] for d in _dims if d in _PHYS)
            if _cl:
                for _ti in range(len(prompt_schedule)):
                    prompt_schedule[_ti] = (prompt_schedule[_ti] + " " + _cl).strip()
        if _os.environ.get("WBENCH_SOLIDITY_CLAUSES", "0") == "1":
            _SOLID = {
                "collision": "All objects are rigid and solid: feet, wheels and props rest firmly on surfaces, and every contact stops cleanly at the surface with visible support.",
                "human_physics": "The character moves with smooth, natural, anatomically coherent motion, limbs articulating fluidly and the face keeping a stable, consistent structure.",
            }
            _dims2 = ((case.get("causal_fidelity") or {}).get("dims") or [])
            _cl2 = " ".join(_SOLID[d] for d in _dims2 if d in _SOLID)
            if _cl2:
                for _ti in range(len(prompt_schedule)):
                    prompt_schedule[_ti] = (prompt_schedule[_ti] + " " + _cl2).strip()
        perspective = str(settings.get("perspective", case.get("perspective", "third_person")))
        return self._assemble(
            image=image, actions=actions, caption=caption, perspective=perspective,
            disp_id=f"case_{case_id}", case_id=str(case_id), case_path=str(case_path),
            extra={
                "wbench_prompt_schedule": prompt_schedule,
                "wbench_turn_actions": _full_turn_actions(case),  # camera action of every turn
                "wbench_subject_mask": subject_mask_path,
            },
        )

    def _getitem_custom(self, entry: dict[str, Any]) -> dict[str, Any]:
        if entry["kind"] == "image":
            image = Image.open(entry["path"]).convert("RGB")
        else:  # video
            image = self._read_first_frame(entry["path"])
        return self._assemble(
            image=image, actions=entry["actions"], caption=entry["caption"],
            perspective=entry["perspective"], disp_id=entry["disp_id"],
            case_id=entry["disp_id"], case_path="",
            extra={"wbench_pose_case_id": entry.get("pose_case_id"), "wbench_first_frame_src": entry["path"]},
        )

    def _assemble(
        self, *, image: Image.Image, actions: list[str], caption: str, perspective: str,
        disp_id: str, case_id: str, case_path: str, extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        caption = _strip_camera(caption)
        if extra and isinstance(extra.get("wbench_prompt_schedule"), list):
            extra = dict(extra)
            extra["wbench_prompt_schedule"] = [_strip_camera(p) for p in extra["wbench_prompt_schedule"]]
        image_tensor = self._process_image(image).sub_(0.5).div_(0.5)
        video_pixels = image_tensor.unsqueeze(0).repeat(self.frames, 1, 1, 1).contiguous()
        intrinsic = torch.tensor(
            [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        cam_c2w = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(self.frames, 1, 1)
        metadata: dict[str, Any] = {
            "intrinsic": intrinsic,
            "cam_c2w": cam_c2w,
            "video_id": disp_id,
            "has_camera": True,
            "source": "wbench_navi",
            "caption_type": "wbench",
            "pose_orig_w": float(image.width),
            "pose_orig_h": float(image.height),
            "frame_start": 0,
            "frame_end": self.frames - 1,
            "intrinsic_raw": intrinsic.clone(),
            "cam_c2w_raw": cam_c2w.clone(),
            "wbench_case_id": str(case_id),
            "wbench_case_path": str(case_path),
            "wbench_perspective": perspective,
            "wbench_actions": actions,
            "wbench_n_turns": int(len(actions)),
            "wbench_environment_prompt": caption,
            "wbench_character_prompt": "",
            "wbench_perspective_prompt": "",
        }
        if extra:
            metadata.update(extra)
        return {"video_pixels": video_pixels, "caption": caption, "metadata": metadata}

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        tensor = self._to_tensor(image)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.height, self.width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return tensor.squeeze(0).contiguous()


def _case_sort_key(path: Path) -> tuple[int, str]:
    raw = path.stem.replace("case_", "")
    try:
        return int(raw), raw
    except ValueError:
        return 10**9, raw


def _is_navigation_case(case: dict[str, Any]) -> bool:
    return _navigation_turn_count(case) > 0


def _navigation_turn_count(case: dict[str, Any]) -> int:
    count = 0
    for item in case.get("interactions", []) or []:
        if str(item.get("type", "")).lower() == "navigation":
            count += 1
    return count


def _navigation_actions(case: dict[str, Any]) -> list[str]:
    return [
        str(item.get("action", "stop"))
        for item in (case.get("interactions", []) or [])
        if str(item.get("type", "")).lower() == "navigation"
    ]


def _max_turn(case: dict[str, Any]) -> int:
    its = case.get("interactions", []) or []
    turns = [int(it.get("turn", 0) or 0) for it in its]
    return max(turns) if turns else 0


def _full_turn_actions(case: dict[str, Any], default: str = "W") -> list[str]:
    """Return the camera action of each turn 1..max_turn (used by the all-turns interaction test):
       - navigation turn -> that turn's navigation action
       - interaction turn -> reuse the previous navigation action so the camera keeps moving while
         the prompt schedule drives the content change.
       For pure navigation cases the result matches the plain navigation actions.
    """
    its = case.get("interactions", []) or []
    nav = {
        int(it.get("turn", 0) or 0): str(it.get("action", "stop"))
        for it in its
        if str(it.get("type", "")).lower() == "navigation"
    }
    out: list[str] = []
    last = default
    for t in range(1, _max_turn(case) + 1):
        if t in nav:
            last = nav[t]
        out.append(last)
    return out


def _read_sekai_description(path: Path) -> str | None:
    try:
        d = json.load(open(path, "r", encoding="utf-8"))
        desc = (d.get("overall", {}) or {}).get("description")
        return str(desc).strip() if desc else None
    except Exception:
        return None


def _read_caption_txt(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not text:
        return None
    marker = "Original Caption:"
    if marker in text:
        text = text.split(marker, 1)[0].strip()
    return " ".join(text.split()) or None


def _build_caption(case: dict[str, Any]) -> str:
    parts = [
        str(case.get("environment_prompt", "")).strip(),
        str(case.get("character_prompt", "")).strip(),
        str(case.get("perspective_prompt", "")).strip(),
    ]
    return " ".join(part for part in parts if part)


def _load_rewritten_prompts(root: Path) -> dict[str, dict[str, str]]:
    """Load the offline training-style prompt rewrites (root/prompts_training_style.jsonl or an env override).
       Returns {case_id: {"scene": ..., "camera": ...}}; an empty dict when the file is absent.
    """
    import os

    path = os.environ.get("WBENCH_REWRITE_JSONL") or str(Path(root) / "prompts_training_style.jsonl")
    out: dict[str, dict[str, str]] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            entry: dict[str, Any] = {
                "scene": str(d.get("scene", "")).strip(),
                "camera": str(d.get("camera", "")).strip(),
            }
            if isinstance(d.get("interactions"), dict):
                entry["interactions"] = {str(k): str(v) for k, v in d["interactions"].items()}
            out[str(d["case_id"])] = entry
    return out


def _build_training_style_caption(
    case: dict[str, Any], *, rewritten: dict[str, str] | None = None,
    scene_extra: str = "", camera_extra: str = "",
) -> str:
    """Assemble a benchmark prompt into the training-data full_prompt style (scene prose plus a <camera> block).
       - when a rewrite exists it is used as the base (preferred path)
       - otherwise fall back to concatenating the three original benchmark fields.
       scene_extra / camera_extra carry interaction additions (events and subject actions go to the scene,
    """
    if rewritten:
        scene_base = str(rewritten.get("scene", "")).strip()
        cam_base = str(rewritten.get("camera", "")).strip()
    else:
        char = str(case.get("character_prompt", "")).strip()
        env = str(case.get("environment_prompt", "")).strip()
        scene_base = " ".join(p for p in [char, env] if p)
        cam_base = str(case.get("perspective_prompt", "")).strip()
    scene = " ".join(p for p in [scene_base, scene_extra.strip()] if p)
    cam = " ".join(p for p in [cam_base, camera_extra.strip()] if p)
    out = scene
    if cam:
        out = f"{scene} <camera>{cam}</camera>" if scene else f"<camera>{cam}</camera>"
    return out.strip()


def _build_interaction_prompt_schedule(
    case: dict[str, Any], *, rewritten: dict[str, str] | None = None
) -> list[str]:
    """Return the training-style prompt per turn, accumulating interactions:
       - event_edit / subject_action.action -> appended to the scene narrative
       - perspective_switch.action -> appended to the <camera> block
       Turns are 1-based; index 0..(max_turn-1) corresponds to turn 1..max_turn.
    """
    its = case.get("interactions", []) or []
    turns = [int(it.get("turn", 0) or 0) for it in its]
    max_turn = max(turns) if turns else 1
    inter_rw = (rewritten or {}).get("interactions") or {}
    scene_acc: list[str] = []
    camera_acc: list[str] = []
    sched: list[str] = []
    for t in range(1, int(max_turn) + 1):
        for it in its:
            if int(it.get("turn", 0) or 0) != t:
                continue
            typ = str(it.get("type", ""))
            act = str(inter_rw.get(str(t)) or it.get("action", "")).strip()
            if not act:
                continue
            if typ in ("event_edit", "subject_action"):
                scene_acc.append(act)
            elif typ == "perspective_switch":
                camera_acc.append(act)
        sched.append(
            _build_training_style_caption(
                case, rewritten=rewritten,
                scene_extra=" ".join(scene_acc), camera_extra=" ".join(camera_acc),
            )
        )
    return sched

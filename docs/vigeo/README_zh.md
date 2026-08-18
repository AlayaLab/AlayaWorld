<p align="right">
  <kbd><a href="README.md">English</a></kbd>
  <kbd><b>简体中文</b></kbd>
</p>

# Alaya-World — 自回归视频世界模型训练代码

以 **LTX-2.3（约 13B）视频 DiT** 为底座、改造成自回归世界模型的训练与推理代码：给定首帧、
文本 prompt 和相机轨迹，模型逐 chunk 滚动生成视频，同时保持长程记忆与三维一致性。

本仓库包含：

- **四个训练阶段**，每个阶段一份 YAML + 一个 launcher —— 双向预训练、历史预训练、
  带几何条件的自回归 SFT、以及蒸馏到 4 步学生模型；
- **缓存预建阶段**（stage0），预建 text-embedding 与整片 VAE latent 两个缓存；
- **推理脚本**：单张图片 + 相机内外参 + prompt → mp4。

```bash
pip install -r requirements.txt

bash scripts/finetune/stage0_precache.sh                                   # 先建缓存
CONFIG_PATH=configs/stage1_pretrain_bidir.yaml bash scripts/finetune/train.sh
```

---

## 1. 安装与准备

### 1.1 Python 环境

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128   # 按你的 CUDA 选
pip install -r requirements.txt
pip install flash-attn --no-build-isolation                                   # FA2,attention 路径需要
```

### 1.2 模型权重

三样东西：LTX-2.3 底座（从它自己的发布渠道获取，本仓不二次分发），加上随本仓发布的两个 checkpoint。

```bash
pip install -U "huggingface_hub[cli]"
export HF_HUB_ENABLE_HF_TRANSFER=1        # 26GB 的 transformer 用它快很多

# 1) LTX-2.3 底座:transformer 与 VAE 在同一个 safetensors 里,外加 text encoder
#    从 LTX-2.3 的发布渠道获取,按下面的表放好

# 2) 自回归 teacher(stage2b):全量微调过的 transformer,25GB
hf download AlayaLab/AlayaWorld-v1.1-stage2b --local-dir weights/alaya-world-ar

# 3) 少步学生(stage3):挂在 teacher 上的 LoRA,2.6GB
hf download AlayaLab/AlayaWorld-v1.1-stage3 --local-dir weights/alaya-world-dmd
```

各仓里有什么、被哪个配置字段吃掉：

| 文件 | 体积 | 配置字段 |
|---|---|---|
| `weights/ltx-2.3/ltx-2.3-22b-dev.safetensors` | 约 50GB | `paths.base_transformer`、`paths.vae`（transformer 与 VAE 同一个文件） |
| `weights/ltx-2.3/google/gemma-3-12b-it-qat-q4_0-unquantized/` | text encoder 目录 | `paths.gemma` |
| `weights/alaya-world-ar/transformer.pt` | 26.2GB | `paths.resume_checkpoint`（填目录） |
| `weights/alaya-world-ar/history_encoder.pt` | 34MB | `paths.history_encoder` |
| `weights/alaya-world-dmd/lora.safetensors` | 2.6GB | `paths.dmd_resume`（填目录） |
| *(critic / GAN 判别器状态)* | — | 发布的 checkpoint 不包含;只有自己跑 stage3 训练才会产生 |

两个推理配置用的都是 teacher 权重；把它从 30 步 teacher 切成 4 步学生的开关就是
`paths.dmd_resume`（见 [推理](#8-推理单图--相机--prompt--视频)）。

### 1.3 ViGeo（stage2b 与 stage3 必需）

自回归阶段的空间条件要调外部 ViGeo pointmap 估计器，本仓不携带。克隆下来，并把 checkpoint 放到
配置期望的位置：

```bash
git clone https://github.com/aigc3d/ViGeo third_party/ViGeo
pip install -r third_party/ViGeo/requirements.txt   # 会装上 xformers,见下面的注意
hf download pkqbajng/ViGeo1.1 --local-dir third_party/ViGeo/checkpoints/ViGeo1.1   # checkpoint 来自 https://huggingface.co/pkqbajng/ViGeo1.1
```

对应配置键：`spatial_memory.vigeo_repo_path`、`spatial_memory.vigeo_checkpoint`，
并且 `spatial_memory.enabled: true`、`context_mode: vigeo_prefix_last_frame`。

> **ViGeo 会 import xformers，而 xformers 与本地编译的 flash-attn-3 不能共存。** xformers 自带一份
> 编译好的 flash-attn-3（`xformers/flash_attn_3/_C.so`），而 PyTorch 规定一个算子命名空间只能有一个
> `TORCH_LIBRARY` 块。两份同时加载会在 C++ 层直接终止进程
> （`Only a single TORCH_LIBRARY can be used to register the namespace flash_attn_3`）——这不是可捕获的
> Python 异常，没有回退余地。所以带 ViGeo/DA3 的阶段一律用 `ALAYA_USE_FA3=0` 跑；FA2 不受影响。

Depth-Anything-3（`spatial_memory.depth_backend: da3`）是另一个深度后端，可选 —— 只有想用那个变体时
才克隆到 `third_party/Depth-Anything-3`。它通过 dinov2 也会引入 xformers，同样受上面这条 FA3 限制。

### 1.4 数据集

数据集分两部分,来自两个地方:

- **标注**(清单 + caption + 相机位姿,打包约 1.5GB)—— 由我们以 **gated(需接受协议)** 的
  Hugging Face 数据集发布:[AlayaLab/AlayaWorld-v1.1-data](https://huggingface.co/datasets/AlayaLab/AlayaWorld-v1.1-data)。
  先在页面接受协议,再用有权限的 token 下载。
- **视频** —— **我们不分发。** 请用上游工具
  [Lixsp11/sekai-codebase](https://github.com/Lixsp11/sekai-codebase)
  下载 Sekai-Real-Walking 源片段(按其 README 拉取 walking 子集),放到
  `data/Video/sekai_real_walking/` 下。全部 18208 条约需 **0.6TB** 磁盘。

```bash
hf auth login                              # 或 export HF_TOKEN=<你的 token>

# 标注:三个文件(清单 jsonl + 两个 tar 包)
hf download AlayaLab/AlayaWorld-v1.1-data --repo-type dataset \
    --local-dir data/Annotation/sekai_real_hq

# 解压 caption/pose,得到配置期望的布局
cd data/Annotation/sekai_real_hq && tar -xzf caption.tar.gz && tar -xzf pose.tar.gz && cd -

# 视频:走 sekai-codebase(见其 README),下到 data/Video/sekai_real_walking/
```

视频只下载一部分也能训 —— 文件缺失的片段会被 loader 跳过。

解开之后,配置期望的布局是:

```
data/Annotation/sekai_real_hq/sekai_real_hq.jsonl      18208 行,每行一条片段
data/Annotation/sekai_real_hq/caption/                 整片 caption + 分段 caption
data/Annotation/sekai_real_hq/pose/*.npz               camera-to-world 外参(估计得到)
data/Video/sekai_real_walking/<video-id>/*.mp4         片段,来自 3879 个源视频(经 sekai-codebase 下载)
```

把 `paths.annotation_base_dir` 指到 `data/Annotation`,`paths.video_base_dir` 指到 `data/Video`。
jsonl 字段与 caption/pose 的 schema 见 [数据格式](#4-数据格式)。

标注的使用受你在数据集页面接受的协议约束。**使用该数据即默认你已遵守
[sekai-codebase](https://github.com/Lixsp11/sekai-codebase) 的数据使用协议**——素材由此派生。
相机位姿是我们自己估计的结果,不是源站标注。

### 1.5 启动前自检

```bash
DESCRIBE=1 CONFIG_PATH=configs/stage1_pretrain_bidir.yaml bash scripts/finetune/train.sh
```

它只解析 config、打印摘要就退出，不碰 GPU —— 权重、数据集或 ViGeo 路径缺失会当场报错。

## 2. 环境要求

| | |
|---|---|
| GPU | launcher 默认单节点 8 卡（`NPROC_PER_NODE`）。全量微调阶段（stage1 / stage2b）需要单卡 ≥80GB 并用 FSDP 分片优化器状态；LoRA 阶段（stage2a / stage3）也可以 `runtime.fsdp: false` 跑 |
| Python / CUDA | Python 3.10、CUDA 12.x |
| PyTorch | >= 2.7.1（`requirements.txt` 记录的是实际可用的下限，未锁上界） |
| Attention | flash-attn 2 即可。flash-attn-3（Hopper）可选，需自行编译，且不能与 ViGeo/DA3 阶段同用，见 [安装与准备](#1-安装与准备) |
| 磁盘 | 模型权重约 80GB（LTX-2.3 底座 + 发布的 AR/DMD checkpoint）；数据集约 0.65TB；544×960 下整片 VAE latent 缓存再约 0.4TB |

`ltx2/` 与 `fastvideo/` 是随仓携带的第三方栈，出处见 `THIRD_PARTY.md`。

## 3. 权重与数据的目录布局

配置文件里所有外部路径都是**相对仓库根目录**的。把文件放进去，或做软链接：

```
weights/ltx-2.3/                LTX-2.3 transformer .safetensors、VAE、text encoder 目录
third_party/ViGeo/              外部 ViGeo 几何估计器（含 checkpoints/ViGeo1.1）
third_party/Depth-Anything-3/   仅 depth-warp 这一空间条件变体需要
data/Video/                     训练视频片段（mp4）
data/Annotation/                每片的 caption json + 相机 pose npz + 数据集 jsonl
cache/                          text-embedding 与 VAE latent 缓存（由 stage0 写入）
outputs/, logs/                 运行产物（已 git-ignore）
```

`ALAYA_DATASET_CACHE_DIR` 可改样本清单缓存的位置（默认 `.cache/dataset`；指向
`/dev/shm/dataset_cache` 可以用内存盘）。

## 4. 数据格式

随仓配置只用一个数据源 `sekai_real_hq`（`data.sources: {sekai_real_hq: 1.0}`），数据集本身不随仓分发。
加载器（`fastvideo/dataset/t2v_datasets.py`）对每条片段要求三样东西。

**jsonl，每行一条片段。** 相对路径分别拼 `paths.video_base_dir` / `paths.annotation_base_dir`，
绝对路径原样使用。

```json
{"video": "sekai_real_hq/clip_0001.mp4",
 "prompt": "sekai_real_hq/clip_0001.json",
 "pose": "sekai_real_hq/clip_0001.npz",
 "num_frames": 1800,
 "valid_k8_starts": [0, 8, 16]}
```

`num_frames` 让采样器不用打开视频就能选窗口；`valid_k{K}_starts` 是可选项，只在
`layout.k8_use_valid_starts` / `k4_use_valid_starts` 打开时读取。

**caption json** —— 一条整片 caption，加可选的分段 caption。训练时以
`data.overall_caption_prob` 的概率取整片 caption，否则取分段 caption：

```json
{"overall":  {"short_prompt": "...", "full_prompt": "..."},
 "segments": [{"time_range_s": [0.0, 4.0], "full_prompt": "..."}]}
```

**相机 pose npz** —— camera-to-world 外参，有内参就一并放进来：

| 键 | 形状 | 说明 |
|---|---|---|
| `cam_c2w`（或 `extrinsic`、`data`） | `[N, 4, 4]` | camera-to-world，每个视频帧一个 |
| `intrinsics` / `intrinsic` / `K` | `[3, 3]` 或 `[N, 3, 3]` | 可选；同目录的内参 npz 会被自动识别，像素单位的值会按画面尺寸归一化 |

外参在使用前会被归一化（`data.camera_norm_mode`、`camera_post_relic_scale`），
使平移量落在 bf16 位置编码安全的数值范围内。

## 5. 训练阶段

| config | 阶段 | trainer / 模式 | 机制 | 上游权重 |
|---|---|---|---|---|
| `configs/stage0_precache.yaml` | **缓存预建（不训练）** | `scripts/finetune/stage0_precache.sh` | 整片 VAE latent 缓存（全卡分片、可断点续建）+ text-embedding 缓存 | LTX-2.3 VAE + text encoder |
| `configs/stage1_pretrain_bidir.yaml` | 双向预训练 | `RolloutTrainer` / sft | 整段 20s 片一次性去噪；片内 clean-mask 条件（i2v 0.7 / v2v 0.2 / t2v 0.1，条件帧在模型内部 sigma=0）；变长训练 | LTX-2.3 base 权重 |
| `configs/stage2a_histpretrain.yaml` | 历史预训练 | `FrameQueryTrainer` / lora | 掩码历史重建：非 Ω 帧加噪、Ω 帧保持 clean → HistoryEncoder → 重建 Ω；训 HistoryEncoder + LoRA（rank 256） | stage1 checkpoint |
| `configs/stage2b_arsft_vigeo.yaml` | 自回归 SFT | `RolloutTrainer` / sft | sink（远端）+ 历史记忆 + nearby motion latent + **ViGeo** pointmap 前缀几何 + 动作 AdaLN；anti-drift 与 next-forcing | stage1 transformer + stage2a HistoryEncoder |
| `configs/stage3_dmd_vigeo.yaml` | 少步蒸馏 | `DmdTrainer` / lora | DMD（TTUR 1:10）+ rCM 一致性正则 + Self-Forcing++ 自滚动；产出 4 步学生 | stage2b checkpoint（teacher 与 critic 共用） |

四个阶段是串起来的：**stage0 → stage1 → stage2a → stage2b → stage3**。后一阶段读前一阶段的产出，
所以下面这些字段要填成你自己的运行目录：

| config 字段 | 指向 |
|---|---|
| `stage2a: paths.resume_checkpoint` | stage1 输出的 checkpoint |
| `stage2b: paths.resume_checkpoint` + `paths.history_encoder` | stage2a 的 checkpoint，且 LoRA 已合并回 base（用 `scripts/tools/merge_lora_for_rollout.py`） |
| `stage3: paths.resume_checkpoint` + `paths.history_encoder` | stage2b 的 checkpoint |
| `stage3: paths.dmd_resume` | stage3 自己的 checkpoint（续训学生时填） |

训练中的 validation 按 `validation.interval` 把视频写到 `outputs/<run>/validation/step-XXXXXX/`；
`validation.before_train: true` 会在第 1 步前先出一版基线。

## 6. Launcher 开关

`scripts/finetune/train.sh` 读 `CONFIG_PATH`，并按 config 里打开了哪个 `*.enabled`
来分派 trainer（见 `alaya/train.py`）：

| 变量 | 作用 |
|---|---|
| `VALIDATE_ONLY=1` | 只跑 validation，不训练 |
| `DESCRIBE=1` | 打印解析后的 config 摘要并退出 |
| `LOG_FILTER=all` | 保留全部 stdout（默认只留 `[Train] step=` 行） |
| `NPROC_PER_NODE` / `MASTER_PORT` | 覆盖 torchrun 拓扑 |
| `ALAYA_USE_FA3=1` + `FA3_HOPPER=<path>` | 使用本地编译的 flash-attn-3（Hopper），loss 与 FA2 逐位一致。**与 ViGeo / DA3 空间条件路径互斥**：那条路径会引入 xformers，而 xformers 自带一份 flash-attn-3 —— 这些阶段用 `ALAYA_USE_FA3=0`（见 [安装与准备](#1-安装与准备)） |
| `LOG_ROOT` / `LOG_NAME` | 覆盖日志落盘位置（默认由 `run.log_dir` 或 config 名派生） |

`runtime.fsdp: false` 只适用于 LoRA 阶段 —— 全量微调阶段必须分片，否则优化器状态放不下。

## 7. 缓存

两个磁盘缓存跨阶段、跨重启共享：

| 缓存 | config 键 | 行为 |
|---|---|---|
| text embedding | `runtime.text_embed_cache_dir` | 懒加载：训练时 miss 就编码并写回。`runtime.precache_text_embeds: true`（stage0）会把所有可达 prompt 提前编码 |
| 整片 VAE latent | `runtime.vae_latent_cache_dir` | **训练期只读** —— 训练不写这个缓存，必须由 stage0 预建（`scripts/tools/precache_vae_latents.py`） |

不做 VAE 预建的话，每步都要重新编码整个窗口；预建之后窗口尾部从缓存切片、只有窗首现算
（长窗阶段实测 VAE 耗时下降约 60%）。

**`[Perf]` 行怎么读**（每 10 步打印一次）：

```
[Perf] last10: text_encode=0.39s vae_encode=5.86s text_cache_hit=10/10 vae_cache_hit=10/10 cache_size=10
```

`vae_cache_hit=0/0` —— 分母是 0，说明缓存一次都没被查 —— 在**短窗阶段是正常的**
（stage2b 和 stage3 用 K=4）。缓存只能提供窗口第 17 个之后的 latent：因果 VAE 需要大约 16 个
latent 的上下文，切片值才与现算逐位一致；4 个 latent 的窗口没有可切片的尾部，于是整个查询被跳过。
这不损失什么 —— 现算 4 个 latent 比命中缓存地现算 61 个还便宜。

## 8. 推理：单图 + 相机 + prompt → 视频

```bash
bash scripts/infer/generate_video.sh \
    --image first_frame.png \
    --prompt "a first-person walk down a misty forest trail" \
    --synth-frames 256 --forward 0.0049 --yaw 0.15 \
    --rounds 5
```

脚本先准备输入（`scripts/infer/prepare_i2v_inputs.py`），再以 validation-only 模式跑模型。
把 `paths.resume_checkpoint` / `paths.history_encoder` 指到你的 stage2b checkpoint，
`paths.dmd_resume` 指到 stage3。

| 输入 | 怎么给 |
|---|---|
| 首帧 | `--image`（任意分辨率，会缩放到 config 的 544×960） |
| prompt | `--prompt` |
| **外参** | `--extrinsics my_c2w.npz`（`cam_c2w` `[N,4,4]`，camera-to-world）**或**用 `--synth-frames/--forward/--yaw/--pitch` 合成轨迹 |
| **内参** | `--intrinsic fx fy cx cy`（归一化值）。不给时，几何后端会自行拟合内参，而不是相信占位值 |
| 长度 | `--rounds N` —— 每轮 4 个 latent（约 1.3s 视频） |

两个 config，同一个脚本：

| config | 模型 | 采样 |
|---|---|---|
| `configs/infer_i2v_camera.yaml` | stage3 的 4 步 DMD 学生（`paths.dmd_resume`） | 4 步、uniform、CFG 1.0 —— 快 |
| `configs/infer_i2v_camera_ar.yaml` | stage2b 的 AR-SFT teacher（不加学生 LoRA） | 30 步、shift、CFG 3.0 —— 慢，质量参照 |

```bash
CONFIG_PATH=configs/infer_i2v_camera_ar.yaml \
    bash scripts/infer/generate_video.sh --image a.png --prompt "..." --rounds 5
```

## 9. 代码结构

```
alaya/            自研训练代码
  train.py        入口 + trainer 分派
  config/         dataclass schema + YAML 加载
  trainer/        RolloutTrainer / FrameQueryTrainer / DmdTrainer / dmd_self_rollout (SF++)
  memory/         HistoryEncoder、ViGeo 几何、DA3 深度、空间缓存
  dmd/            DMD loss、rCM 一致性、next-forcing、判别器
  model/          LTX 加载、LoRA、FSDP 包装
  data/           dataloader、text-embedding 与 VAE latent 缓存
configs/          stage0-stage3 训练配置 + 两个推理配置
scripts/finetune/ train.sh launcher、stage0_precache.sh
scripts/infer/    单条数据推理入口
scripts/tools/    VAE latent 预建、rollout 用的 LoRA 合并
ltx2/, fastvideo/ 随仓携带的第三方栈（见 THIRD_PARTY.md）
```

## 10. 已知现象

| 现象 | 解释 |
|---|---|
| stage2b / stage3 上 `vae_cache_hit=0/0` | 正常，见 [缓存](#7-缓存) |
| stage3 出现 `RuntimeError: sample too short` 重试 | Self-Forcing++ 窗口需要 `81 + max_chunks*K*stride` 帧，短于此的片段会被跳过重采，属于 dataset 正常行为，不是报错 |
| 进程直接终止，报 `Only a single TORCH_LIBRARY can be used to register the namespace flash_attn_3` | 本地编译的 flash-attn-3 与 xformers 内置的那份（ViGeo 直接引入，DA3 经 dinov2 引入）注册了同一个 torch 算子命名空间。这是 C++ 层 abort，捕获不到也无法回退 —— 所有空间条件阶段都要设 `ALAYA_USE_FA3=0` |
| stage1 / stage2b 配 `runtime.fsdp: false` 时 OOM | 全量微调阶段需要 FSDP 分片优化器状态 |

## 11. 许可与第三方代码

本仓库的许可见 `LICENSE`；随仓携带的 `ltx2/`、`fastvideo/`，Helios 衍生的 anti-drift 与 GAN 配方，
以及运行时加载但不随仓分发的外部估计器（ViGeo、Depth-Anything-3）的出处与许可见 `THIRD_PARTY.md`。

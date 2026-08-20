# AlayaWorld:长时程可交互视频世界生成

<p align="center"><a href="https://alayalab.ai/"><b>Alaya Lab</b></a></p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-e5e7eb?style=for-the-badge"></a>
  <a href="README_zh.md"><img src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-2563eb?style=for-the-badge"></a>
</p>

<p align="center">
  <a href="https://alaya-lab.github.io/AlayaWorld/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
  <a href="https://www.youtube.com/watch?v=n0jIEg7taTI"><img src="https://img.shields.io/badge/YouTube-Demo-red?logo=youtube&logoColor=white"></a>
  <a href="https://arxiv.org/abs/2607.06291"><img src="https://img.shields.io/badge/Intro-Report-red"></a>
  <a href="https://arxiv.org/abs/2607.18367"><img src="https://img.shields.io/badge/Full-Report-red"></a>
  <a href="https://arxiv.org/abs/2608.13492"><img src="https://img.shields.io/badge/v1.1-Report-red"></a>
  <a href="https://github.com/AlayaLab/AlayaWorld"><img src="https://img.shields.io/badge/Code-Available-brightgreen?logo=github"></a>
  <a href="#weights"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HuggingFace-yellow"></a>
</p>

<p align="center">
  <img src="assets/fig1-AlayaWorld.png" width="100%">
</p>

> 一个可交互的自回归世界模型,支持实时相机控制、提示词切换,以及长时程记忆一致性。

---

## 📰 最新动态

- **[2026-08-20]** 支持**交互式浏览器 demo**:实时游玩 AlayaWorld——键盘开车、边生成边改 prompt、边生成边播放。见 [`reactor/`](reactor/README.md)。特别感谢社区贡献者 [@Dere-Wah](https://github.com/Dere-Wah) 与 [@Rising0321](https://github.com/Rising0321)!
- **[2026-08-17]** 开源**全栈训练+推理代码**、**v1.1 权重**(AR + DMD)与**部分训练数据**,同步发布 [v1.1 技术报告](https://arxiv.org/abs/2608.13492)。见[发布路线图](#-发布路线图)。
- **[2026-07-21]** 发布[完整技术报告](https://arxiv.org/abs/2607.18367)。
- **[2026-07-16]** 发布推理代码,预训练权重已上线 🤗 [Hugging Face](https://huggingface.co/AlayaLab/AlayaWorld)。参见[快速开始](#-快速开始)。
- **[2026-07-08]** 发布项目主页与[技术报告](https://arxiv.org/abs/2607.06291)。

## 🚀 发布路线图

- [x] 推理代码
- [x] 预训练权重 — 🤗 [AlayaLab/AlayaWorld](https://huggingface.co/AlayaLab/AlayaWorld)
- [x] 预训练权重 v1.1 — AR:🤗 [AlayaWorld-v1.1-stage2b](https://huggingface.co/AlayaLab/AlayaWorld-v1.1-stage2b) · DMD:🤗 [AlayaWorld-v1.1-stage3](https://huggingface.co/AlayaLab/AlayaWorld-v1.1-stage3)
- [x] 训练代码
- [x] 训练数据(部分)— 🤗 [AlayaWorld-v1.1-data](https://huggingface.co/datasets/AlayaLab/AlayaWorld-v1.1-data)

## ✨ 核心特性

AlayaWorld 围绕四大核心特性构建 —— **交互性**、**一致性**、**稳定性** 与 **实时性**。

### 🎮 交互性
两条控制通道:一条是渲染的 3D 缓存配合轻量级 AdaLN 相机调制,实现有据可依、贴合轨迹的导航;另一条是 chunk 级别的提示词切换,可在生成过程中引入新事件。

### 🧠 一致性
两种互补的记忆形式:一是可显式重投影到查询视角的 3D 缓存,用于空间召回;二是压缩后的帧历史嵌入,用于时间连续性 —— 从而让重访过的场景保持可辨认。

### 🛡️ 稳定性
长时程稳定性来自在"漂移历史"上训练,以及一个误差库(error bank):它把累积的伪影重新注入记忆与目标,防止误差在长达数分钟的 rollout 中不断叠加。

### ⚡ 实时性
通过少步 DMD 蒸馏与短时间 chunk 实现实时交互,并在 chunk 边界处切换提示词,以同时把视觉与语义延迟降到最低。

## 📁 仓库结构

```
alaya/          世界模型核心:config / data / memory / model / 训练器
                └── inference/   da3 流式 case-demo 推理管线
ltx2/           LTX-2.3 模型栈(DiT / VAE / 文本编码器 / 相机控制)
fastvideo/      训练共用的数据集与 rollout 工具
scripts/        finetune/train.sh(统一启动器)· infer/ 辅助 · tools/
configs/        stage0–stage3 训练 + 三条推理配置
inference/      da3 case-demo 命令行入口(run.sh / run.py)
reactor/        把 da3 路径变成可实时游玩的直播流(含浏览器 demo)
playground/     内置演示用例(case1)
docs/vigeo/     完整训练手册(数据格式、各阶段、启动器参数)
```

一个启动器驱动一切——训练、验证、三条推理路,全部由配置选择:

```bash
CONFIG_PATH=configs/<任意>.yaml bash scripts/finetune/train.sh
```

## 🏃 快速开始

**1. 环境** —— 一个 Python 环境覆盖训练与全部推理(在 torch 2.7.1 + CUDA 12.8 上验证):

```bash
pip install -r requirements.txt
# Depth-Anything-3(da3 推理路径)是代码仓库,在上述依赖之后安装:
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 third_party/Depth-Anything-3
pip install -e third_party/Depth-Anything-3
```

<a id="weights"></a>
**2. 权重**

| 组件 | 用于 | 来源 |
|---|---|---|
| `merged_infer.safetensors` — DiT+VAE+文本编码器+历史编码器 打包 | da3 推理 | 🤗 [AlayaLab/AlayaWorld](https://huggingface.co/AlayaLab/AlayaWorld) |
| LTX-2.3 底座(`ltx-2.3-22b-dev.safetensors`)| 训练、AR/DMD 推理 | 🤗 [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) |
| AR teacher v1.1(stage2b,完整 transformer)| AR 推理、stage3 训练 | 🤗 [AlayaWorld-v1.1-stage2b](https://huggingface.co/AlayaLab/AlayaWorld-v1.1-stage2b) |
| 少步 student v1.1(stage3 LoRA)| DMD 推理 | 🤗 [AlayaWorld-v1.1-stage3](https://huggingface.co/AlayaLab/AlayaWorld-v1.1-stage3) |
| Gemma 文本编码器 | 全部 | 🤗 [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)(受限)|
| ViGeo checkpoint(ViGeo1.1)| 训练、AR/DMD 推理 | 🤗 [pkqbajng/ViGeo1.1](https://huggingface.co/pkqbajng/ViGeo1.1)(代码:[aigc3d/ViGeo](https://github.com/aigc3d/ViGeo))|
| Depth-Anything-3 权重 | da3 推理 | 🤗 [depth-anything/DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) |

权重路径都在各配置的 `paths:` 下——按你的下载位置改。完整目录规划见
[`docs/vigeo/README.md`](docs/vigeo/README.md)。

**3. 推理** —— 三条路,同一种启动形式:

```bash
# a) case demo(da3 空间记忆):首帧图 + camera.pt + prompt -> 约 1 分钟视频
CONFIG_PATH=configs/infer.yaml bash scripts/finetune/train.sh
#    等价的命令行捷径(可带 seed / rounds / ttc / skill 等单次参数):
bash inference/run.sh                     # 内置 playground/case1
python -m inference.run --input playground/case1/case1 --seed 1234 --rounds 45

# b) 自回归 teacher,30 步(vigeo 空间记忆)
VALIDATE_ONLY=1 CONFIG_PATH=configs/infer_i2v_camera_ar.yaml bash scripts/finetune/train.sh

# c) 少步 student,4 步(vigeo 空间记忆)
VALIDATE_ONLY=1 CONFIG_PATH=configs/infer_i2v_camera.yaml bash scripts/finetune/train.sh
```

以上三条都是按固定轨迹渲染成文件。想边生成边开车、随时改 prompt,
[`reactor/`](reactor/README.md) 把 a) 这条路径做成了实时流并配了浏览器 demo:

```bash
reactor build -f Dockerfile.reactor
reactor run --gpus device=0 -e HF_TOKEN     # 然后进 reactor/demo 起前端
```

说明:b/c 走训练器的验证循环,必须带 `VALIDATE_ONLY=1`;a) 由配置里的
`da3_infer.enabled` 直接分派进推理管线,该变量对它无效。vigeo 空间路径与 FA3
不兼容——如果你编译了 FA3,启动时加 `ALAYA_USE_FA3=0`。

a) 的完整参数列表见 [`inference/README_zh.md`](inference/README_zh.md)。

b/c 想用自己的图+轨迹:`scripts/infer/generate_video.sh` 一步完成输入准备
(`--image/--prompt` 配 `--extrinsics` 或 `--synth-frames`)并启动 c)。
a) 的用例是共享前缀的三个文件:

```
<prefix>_image.png     首帧(初始化历史)
<prefix>_camera.pt     相机轨迹:cam_c2w [F,4,4] + 内参
<prefix>_prompt.txt    文本提示
<prefix>_skill.txt     (可选)最后几秒的提示词(一次性收尾特效)
```

**4. 训练** —— 四个阶段,同一启动器;每阶段的 `paths:` 指向上一阶段的 checkpoint:

```bash
CONFIG_PATH=configs/stage0_precache.yaml      bash scripts/finetune/train.sh  # VAE latent + 文本嵌入缓存
CONFIG_PATH=configs/stage1_pretrain_bidir.yaml bash scripts/finetune/train.sh # 双向预训练
CONFIG_PATH=configs/stage2a_histpretrain.yaml bash scripts/finetune/train.sh  # 历史编码器预训练
CONFIG_PATH=configs/stage2b_arsft_vigeo.yaml  bash scripts/finetune/train.sh  # 自回归 SFT(ViGeo)
CONFIG_PATH=configs/stage3_dmd_vigeo.yaml     bash scripts/finetune/train.sh  # 少步 DMD 蒸馏
```

数据下载、数据格式、各阶段细节与启动器参数:[`docs/vigeo/README.md`](docs/vigeo/README.md)。

## 👥 团队

- **核心负责人:** Kaipeng Zhang
- **负责人:** Chuanhao Li
- **核心贡献者:** Chuanhao Li、Kaipeng Zhang、Yifan Zhan、Yongtao Ge、Yuanyang Yin
- **贡献者:** Jiaming Tan、Kang He、Liaoyuan Fan、Mingliang Zhai、Ruicong Liu、Xiaojie Xu、Xuangeng Chu、Zhen Li、Zhengyuan Lin、Zhixiang Wang、Zian Meng、Zihui Gao

## 📬 联系我们

如需合作或商务咨询,请联系 **kaipeng.zhang@shanda.com**。

## 📝 引用

如果 AlayaWorld 对你的研究有帮助,欢迎引用:

```bibtex
@article{team2026alayaworldintro,
  title={AlayaWorld: Long-Horizon and Playable Video World Generation},
  author={Team, AlayaWorld and Zhang, Kaipeng and Li, Chuanhao and Zhan, Yifan and Ge, Yongtao and Yin, Yuanyang and Tan, Jiaming and He, Kang and Fan, Liaoyuan and Liu, Ruicong and others},
  journal={arXiv preprint arXiv:2607.06291},
  year={2026}
}

@article{team2026alayaworldfull,
  title={AlayaWorld: Long-Horizon and Playable Video World Generation},
  author={Team, AlayaWorld and Zhang, Kaipeng and Li, Chuanhao and Zhan, Yifan and Ge, Yongtao and Yin, Yuanyang and Tan, Jiaming and He, Kang and Fan, Liaoyuan and Liu, Ruicong and Zhai, Mingliang and others},
  journal={arXiv preprint arXiv:2607.18367},
  year={2026}
}
@article{team2026alayaworldv11,
  title={AlayaWorld v1.1: Long-Horizon and Playable Video World Generation},
  author={Team, AlayaWorld and Zhang, Kaipeng and Li, Chuanhao and Zhan, Yifan and Ge, Yongtao and Yin, Yuanyang and Tan, Jiaming and He, Kang and Fan, Liaoyuan and Liu, Ruicong and Zhai, Mingliang and others},
  journal={arXiv preprint arXiv:2608.13492},
  year={2026}
}
```

## 📄 许可证

本项目基于 Lightricks Ltd. 的 LTX-2 构建。原始 LTX-2 代码库的部分内容(`ltx2/`)已由 Alaya Lab 修改,仅供学术与研究用途;所发布的权重(`merged_infer.safetensors`)是从 LTX-2.3 微调而来。因此,本项目 —— 代码与权重 —— 依据 [**LTX-2 社区许可协议(LTX-2 Community License Agreement)**](LICENSE) 发布。LTX-2 的所有原始版权、许可、专利、商标及署名声明均予以保留。

**仅供学术研究与非商业用途。** 如需将 LTX-2 或其衍生物用于商业用途,请联系 Lightricks Ltd.(年营收 ≥ 1000 万美元的主体需获取商业许可)。

完整署名信息见 [NOTICE](NOTICE) 与 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。第三方权重(Gemma-3 文本编码器;Depth-Anything-3)**未在此重新分发** —— 请从其原始来源、依据各自许可协议获取。

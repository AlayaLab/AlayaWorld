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
  <a href="https://github.com/AlayaLab/AlayaWorld"><img src="https://img.shields.io/badge/Code-Available-brightgreen?logo=github"></a>
  <a href="https://huggingface.co/AlayaLab/AlayaWorld"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HuggingFace-yellow"></a>
</p>

<p align="center">
  <img src="assets/fig1-AlayaWorld.png" width="100%">
</p>

> 一个可交互的自回归世界模型,支持实时相机控制、提示词切换,以及长时程记忆一致性。

---

## 📰 最新动态

- **[2026-07-21]** 发布[完整技术报告](https://arxiv.org/abs/2607.18367)。
- **[2026-07-16]** 发布推理代码,预训练权重已上线 🤗 [Hugging Face](https://huggingface.co/AlayaLab/AlayaWorld)。参见[快速开始](#-快速开始)。
- **[2026-07-08]** 发布项目主页与[技术报告](https://arxiv.org/abs/2607.06291)。

## 🚀 发布路线图

- [x] 推理代码
- [x] 预训练权重 — 🤗 [AlayaLab/AlayaWorld](https://huggingface.co/AlayaLab/AlayaWorld)
- [ ] 预训练权重(改进版)
- [ ] 训练代码
- [ ] 训练数据(部分)

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

## 🏃 快速开始

推理为图生视频(image-to-video):给模型一张**首帧图像**、一条**相机轨迹**和一段**文本提示**,它就会沿相机路径逐 chunk 地展开视频(1 chunk ≈ 1.33 秒 @ 24fps;约 45 chunk ≈ 1 分钟)。

**1. 环境** —— 一块 CUDA GPU,以及 PyTorch ≥ 2.6(DiT 使用了 `flex_attention`)。

**2. 权重** —— 模型由四部分组成。只有 `merged_infer.safetensors`(AlayaWorld 自有权重)托管在我们的 🤗 仓库;文本编码器与深度模型是第三方组件 —— 请从各自的原始来源获取:

| `checkpoints/` 下的路径 | 来源 |
|---|---|
| `merged_infer.safetensors` — DiT + VAE + 文本编码器 + 历史编码器 打包 | 🤗 [AlayaLab/AlayaWorld](https://huggingface.co/AlayaLab/AlayaWorld) |
| `gemma-3-12b-it-qat-q4_0-unquantized/` — Gemma 文本编码器 | 🤗 [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)(受限访问 —— 需先接受 Google 的许可协议) |
| `Depth-Anything-3/` — DA3 代码仓库 | GitHub [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)(见第 3 步) |
| `hf_cache/` — DA3 权重,采用 HF-cache 目录结构 | 🤗 [depth-anything/DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1)(见第 3 步) |
| `taeltx2_3_wide.pth` — *可选* TAEHV bank 解码器(`--bank-taehv`) | GitHub [madebyollin/taehv](https://github.com/madebyollin/taehv) |

在仓库根目录下:

```bash
# AlayaWorld 权重(本仓库)
hf download AlayaLab/AlayaWorld merged_infer.safetensors --local-dir checkpoints

# Gemma 文本编码器(受限:需先在其 HF 页面登录并接受许可协议)
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
```

目标目录结构(`configs/infer.yaml` 中的 `paths:` 指向此处;若你的权重存放在别处,请相应修改):

```
checkpoints/
├── merged_infer.safetensors
├── gemma-3-12b-it-qat-q4_0-unquantized/
├── Depth-Anything-3/          # 第 3 步
├── hf_cache/                  # 第 3 步
└── taeltx2_3_wide.pth         # 可选
```

**3. Depth-Anything-3(必需)** —— 空间记忆分支依赖 DA3(一个外部仓库),缺少它推理会直接报错:

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 checkpoints/Depth-Anything-3
pip install -e checkpoints/Depth-Anything-3      # 请在安装完 torch 相关依赖后再执行(见 requirements.txt)
```

其权重(`depth-anything/DA3NESTED-GIANT-LARGE-1.1`)会从 `checkpoints/hf_cache/` 加载 —— 你可以让它在首次运行时自动下载到该目录,或提前拉取:

```bash
HF_HOME=checkpoints/hf_cache hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1
```

**4. 运行一个现成用例**(用例位于 [`playground/`](playground) 下)。
一键启动脚本会渲染内置的 **case1**(约 1 分钟):

```bash
# 单卡
bash inference/run.sh

# 多卡(Ulysses 上下文并行;例如 2 卡或 4 卡)
GPUS=4 bash inference/run.sh
```

`run.sh` 只是转发到 `python -m inference.run`(默认 `--input playground/case1/case1`);直接调用该模块即可运行任意用例或传入额外参数:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m inference.run --input playground/case1/case1 --seed 1234
```

生成的 mp4 会写入 `outputs/` 下。完整参数列表见 [`inference/README.md`](inference/README.md)。

**5. 使用你自己的输入** —— 一个"用例"由共享前缀的三个文件组成:

```
<prefix>_image.png     首帧(用于初始化历史)
<prefix>_camera.pt     相机轨迹:cam_c2w [F,4,4] + 内参
<prefix>_prompt.txt    文本提示
```

把 `--input` 指向该前缀即可。对于长时程(约 1 分钟)的 rollout,可加上 `--ttc` 来抑制外观漂移。完整参数列表见 [`inference/README.md`](inference/README.md)。

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
```

## 📄 许可证

本项目基于 Lightricks Ltd. 的 LTX-2 构建。原始 LTX-2 代码库的部分内容(`flash_alaya/ltx2/`)已由 Alaya Lab 修改,仅供学术与研究用途;所发布的权重(`merged_infer.safetensors`)是从 LTX-2.3 微调而来。因此,本项目 —— 代码与权重 —— 依据 [**LTX-2 社区许可协议(LTX-2 Community License Agreement)**](LICENSE) 发布。LTX-2 的所有原始版权、许可、专利、商标及署名声明均予以保留。

**仅供学术研究与非商业用途。** 如需将 LTX-2 或其衍生物用于商业用途,请联系 Lightricks Ltd.(年营收 ≥ 1000 万美元的主体需获取商业许可)。

完整署名信息见 [NOTICE](NOTICE) 与 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。第三方权重(Gemma-3 文本编码器;Depth-Anything-3)**未在此重新分发** —— 请从其原始来源、依据各自许可协议获取。

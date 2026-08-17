<p align="right">
  <kbd><a href="README.md">English</a></kbd>
  <kbd><b>简体中文</b></kbd>
</p>

# Alaya World — 推理(图生视频)

**Alaya World**(基于 LTX 2.3 的自回归可交互世界模型)的官方推理入口。给它一张**首帧图像**、一条**相机/动作轨迹**和一段**文本提示**,它会沿相机路径逐 chunk 地展开视频。

> 1 chunk = 4 个 latent 帧 = 32 个像素帧 ≈ **1.33 秒 @ 24fps**。
> 约 1 分钟的视频 ≈ **45 chunk**,需要 **≥ 约 1450 帧**的相机轨迹。

## 环境要求

- 一块 CUDA GPU。PyTorch **≥ 2.7.1**(DiT 使用 `flex_attention`)。
- 模型权重放在 `./checkpoints/` 下(相对仓库根目录),由 `configs/infer.yaml` 的 `paths:` 指定:

  ```
  checkpoints/
  ├── merged_infer.safetensors                    # DiT + VAE + 文本编码器 + history_encoder 打包
  ├── gemma-3-12b-it-qat-q4_0-unquantized/         # Gemma 文本编码器
  ├── Depth-Anything-3/                            # DA3 代码仓库(空间记忆深度)
  ├── hf_cache/                                    # 存放 DA3 权重的 HF cache
  └── taeltx2_3_wide.pth                           # 可选 tiny bank 解码器(仅 --bank-taehv 时用),来自 github.com/madebyollin/taehv
  ```

  权重在别处就把 `paths:` 改过去。

## 输入格式(一个"用例")

```
<prefix>_image.<png|jpg|jpeg|webp|bmp>   首帧 —— 初始化历史
<prefix>_camera.pt                       metadata 字典:cam_c2w [F,4,4]、intrinsic 等
<prefix>_prompt.txt                      文本提示
```

`--input` 可以指向前缀,也可以指向其中任意一个文件。图像会自动缩放+中心裁剪到配置分辨率(默认 **544×960**),并复制到轨迹长度来初始化模型的历史窗口(模型需要约 5.4 秒历史才能起步)。现成用例在 [`playground/`](../playground) 下。

## 运行

一条命令(单卡)—— 渲染内置的 **case1**(约 1 分钟):

```bash
bash inference/run.sh
```

多卡(Ulysses 上下文并行;如 2 卡或 4 卡):

```bash
GPUS=4 bash inference/run.sh
```

启动脚本只是转发到 `python -m inference.run`(默认 `--input playground/case1/case1`);直接调用该模块可跑任意用例:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m inference.run --input playground/case1/case1 --seed 1234
```

输出:`outputs/<input>_rounds-N.mp4`(用 `--output-dir` 改位置)。默认会叠加 Move/Rotate 摇杆 HUD,用 `--no-joystick` 关闭。当用例带 `<prefix>_skill.txt` 时,视频最后几秒会释放一次性技能特效(用 `--skill-sec 0` 关闭)。

## 常用参数

| 参数 | 默认 | 含义 |
|------|------|------|
| `--input` | *(必填)* | 用例前缀或其中任一文件 |
| `--cfg` | `configs/infer.yaml` | 推理配置;模型路径在 `paths:` 下 |
| `--output-dir` | *(配置)* | mp4 保存位置 |
| `--rounds` | `1000` | 最大自回归 chunk 数;实际 = `min(此值, 轨迹长度)`。约 45 ≈ 1 分钟 |
| `--seed` | `None` | 固定每 chunk 噪声,可复现 |
| `--compile` | `reduce-overhead` | DiT 的 `torch.compile` 模式(`none` 关闭)|
| `--no-flex-attn` | *(默认开)* | 关闭融合 `flex_attention` |
| `--no-joystick` | *(配置)* | 不绘制摇杆 HUD |
| `--ttc` | *(默认关)* | Pathwise Test-Time Correction —— 抑制长视频外观漂移 |
| `--video-crf` | `28` | h264 质量(18 接近无损,28 文件小)|
| `--skill-sec` | `4.0` | 最后 N 秒切换到用例的 `_skill.txt` 提示(一次性收尾特效);`0` 关闭 |
| `--skill-prompt` | *(文件)* | 行内技能提示,覆盖 `<prefix>_skill.txt` |

完整列表见 `python -m inference.run --help`。

## 说明

- 本 CLI 是 `alaya.inference` 引擎的薄封装 —— 原样复用引擎的 rollout 辅助与流式管线。同一次运行也有统一 config 形式:`CONFIG_PATH=configs/infer.yaml bash scripts/finetune/train.sh`(参数在 `da3_infer:` 下)。
- 长(约 1 分钟)rollout 时,`--ttc` 把每个 chunk 重新锚定到首帧以减少外观/风格漂移;其旋钮在配置的 `validation.ttc` 下。

# Manchu Degradation Simulator

用于生成满文古籍图像退化样本的轻量代码仓库，方便单独复用、演示和后续扩展。

这个仓库从原始项目中整理出“退化模拟”相关核心代码，聚焦以下能力：

- 纸张背景与文字前景融合
- 基础图像退化：模糊、JPEG 压缩、亮度对比变化
- 文本区域退化：文字缺损、椒盐式掉墨
- 高级文字损伤：不规则遮挡、墨迹式破坏
- 小样本数据集抽样脚本

## Repository Structure

```text
.
├─ src/
│  ├─ __init__.py
│  ├─ degradation_functions.py
│  ├─ text_degradations.py
│  ├─ advanced_text_damage.py
│  └─ advanced_degradations.py
├─ run_demo.py
├─ sample_manchu_dataset.py
├─ requirements.txt
└─ README.md
```

## Environment

建议使用 Python 3.10+

```bash
pip install -r requirements.txt
```

## Quick Start

准备两张图片：

- 一张干净的满文文字图
- 一张纸张或古籍背景纹理图

运行：

```bash
python run_demo.py \
  --clean path/to/clean.png \
  --background path/to/background.png \
  --output demo_output
```

输出目录中会生成：

- `clean.png`
- `background.png`
- `fused.png`
- `blur.png`
- `jpeg_artifacts.png`
- `brightness_contrast.png`
- `pepper_text_dropout.png`
- `advanced_text_damage.png`

仓库内也附带了少量静态示例图，位于 `examples/`：

- `examples/clean.png`
- `examples/fused.png`
- `examples/pepper_text_dropout.png`
- `examples/advanced_text_damage.png`

## Core Modules

### `src/degradation_functions.py`

基础图像层退化函数：

- Gaussian blur
- JPEG artifacts
- brightness / contrast perturbation

### `src/text_degradations.py`

针对文本区域的退化逻辑：

- 从干净图中提取文本 mask
- 对文字区域做随机掉墨

### `src/advanced_text_damage.py`

更复杂的不规则文字损伤模拟：

- 基于 OpenSimplex 生成破损 mask
- 按文字 mask 与背景 patch 进行局部替换

### `src/advanced_degradations.py`

补充型高级退化逻辑，包含更复杂的局部纸张损坏与文字损坏实现。

## Dataset Sampling

如果你已经有完整的 `manchu_dataset`，可以用下面的脚本快速抽一个小数据集：

```bash
python sample_manchu_dataset.py \
  --src path/to/manchu_dataset \
  --dst path/to/manchu_dataset_sample \
  --num 20 \
  --seed 42 \
  --overwrite
```

## Notes

- 当前仓库保留的是“退化模拟核心代码”，没有打包 OCR、训练或大体量实验输出。
- 原始代码里部分注释是历史编码产物，这不影响运行；后续如果要长期维护，建议逐步统一为 UTF-8 中文或英文注释。
- `advanced_text_damage.py` 与 `advanced_degradations.py` 依赖 `opensimplex` 和 `perlin-noise`。

## Suggested Next Steps

- 将各类退化封装成统一 CLI
- 增加配置文件驱动的批量生成入口
- 补充少量可公开示例图
- 为每种退化输出配套 mask，方便后续训练修复模型

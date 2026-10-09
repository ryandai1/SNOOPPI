# SNOOPPI

[English](../../README.md) | **简体中文**

本仓库提供用于蛋白质–蛋白质相互作用（PPI）二分类的研究笔记本，使用冻结的 **ProtT5、ESM-2 和 ESM-C** 表征，以及对称多层感知机（MLP）分类器。

本文件是主 README 的简体中文版本，覆盖主要实验流程。**英文仍是项目的主要文档语言**；更详细的实验草案、产物说明和复现文档目前使用英文。

[数据集](https://huggingface.co/datasets/ChatterjeeLab/SNOOPPI) · [论文](https://doi.org/10.1093/bioinformatics/btag736) · [复现说明（英文）](../reference/reproducibility.md)

> **实验状态：** 数据、模型权重、表征和实验输出均存放在仓库之外。已保存的笔记本输出已清除；本次整理版**尚未重新运行**以生成结果。

## 目录

- [从这里开始](#从这里开始)
- [仓库导览](#仓库导览)
- [笔记本顺序](#笔记本顺序)
- [运行实验](#运行实验)
- [代码所实现的方法](#代码所实现的方法)
- [理解评估指标](#理解评估指标)
- [可复现性与局限](#可复现性与局限)
- [引用 SNOOPPI](#引用-snooppi)
- [上游资源与署名](#上游资源与署名)

## 从这里开始

| 我想要…… | 从这里开始 |
| --- | --- |
| 运行主要实验 | 按[笔记本顺序](#笔记本顺序)和[运行实验](#运行实验)操作，准备带 GPU 的 Google Colab、已挂载的 Google Drive，以及共享的数据划分 CSV。 |
| 审计已有划分或比较缓存表征 | [实验草案指南（英文）](../guides/draft-experiments.md)。 |
| 审查未知蛋白质对的候选结果 | [挖掘产物指南（英文）](../guides/unknown-pair-mining.md)。模型低分不能确认某个蛋白质对为负例。 |
| 查看主要英文文档 | [English README](../../README.md)。 |

## 仓库导览

| 位置 | 内容 |
| --- | --- |
| [`notebooks/`](../../notebooks/README.md) | ProtT5、ESM-2 和 ESM-C 的三个主要实验。 |
| [`drafting_code/`](../../drafting_code/README.md) | 审计、基线、消融、统计、可解释性和未知蛋白质对挖掘的实验草案，以及共享辅助模块。 |
| [`mined/`](../../mined/README.md) | 挖掘产物指南；仓库中的这个文件夹是目录框架，不是已生成的挖掘数据集。 |
| [`docs/`](../README.md) | 使用指南、参考说明、翻译文档和笔记本整理清单。 |
| [`scripts/`](../../scripts/README.md) | 笔记本验证工具和合成数据冒烟检查。 |
| [`tests/`](../../tests/README.md) | 实验草案协议和未知蛋白质对挖掘的测试。 |

## 笔记本顺序

| 顺序 | 笔记本 | 用途 |
| --- | --- | --- |
| 01 | [ProtT5 预测器](../../notebooks/01_prott5_predictor.ipynb) | 加载带标签的 SNOOPPI 蛋白质对，建立共享数据划分，计算蛋白质表征，并训练和评估分类器。 |
| 02 | [ESM-2 基准实验](../../notebooks/02_esm2_benchmark.ipynb) | 加载已有的数据划分 CSV，评估 ESM-2 表征。 |
| 03 | [ESM-C 基准实验](../../notebooks/03_esmc_benchmark.ipynb) | 加载相同的数据划分文件，审计原始序列和蛋白质对的重叠，并支持表征计算及训练轮次级别的恢复。 |

每个笔记本通过编号章节组织环境设置、数据、表征、训练、评估和导出。历史、可选和恢复章节均在原位置标注。这些笔记本仍是独立实验，没有合并为同一个运行环境。

## 运行实验

1. 在**启用 GPU 的 Google Colab** 中打开笔记本。每种编码器使用新的运行时，并执行各自原有的依赖安装单元。本仓库的 ESM-2 笔记本使用 Hugging Face Transformers；ESM-C 使用 EvolutionaryScale 的 `esm` 包，并有独立的版本约束。
2. 挂载 Google Drive，写入产物前确认 `/content/drive/MyDrive` 确实已挂载。默认实验目录为 `/content/drive/MyDrive/SNOOPPI`。
3. 如果已有原始数据划分 CSV，请保留这些文件，并从笔记本 02 或 03 开始。否则，使用笔记本 01 创建划分。从更新且未固定版本的数据集重新划分，未必能复现原始实验。
4. 对于文档所述的单次固定种子 ProtT5 实验，执行 **03–08** 节，跳过 **09** 节中的历史首次训练循环，再执行 **10–12** 节。第 03 节需要 `datasets` 包；如有需要，通过第 01 节的原始安装单元安装。**01–02** 和 **13–17** 节包含探索、诊断、归档和恢复／导出工具，执行前请阅读各节说明。
5. 在笔记本 02 和 03 中，按编号执行主要章节，并使用同样的三个 CSV。开始或恢复实验前，请检查缓存和检查点路径。保留的原始单元可能覆盖已有实验产物。
6. 为每次实验保留指标、划分文件哈希、实际使用的软件包／模型／数据集版本和配置。确认评估使用的蛋白质对与指标定义一致后，再比较结果。

### 共享输入文件

以下路径相对于 Drive 上的实验目录，而非本仓库。

| 文件 | 用途 |
| --- | --- |
| `splits/train.csv` | 分类器训练。 |
| `splits/validation.csv` | 检查点和分类阈值选择。 |
| `splits/test.csv` | 留出评估。 |

必需的输入列为 `Partner_A_sequence`、`Partner_B_sequence` 和二分类 `label`；基准实验的加载器也支持从 `SNOOPPI_final_label` 推导标签。使用新数据集前，请检查对应加载器的验证逻辑。在基准实验笔记本中，已有的数值标签优先使用。

### 扩展流程

[实验草案套件（英文）](../guides/draft-experiments.md)提供共享划分／缓存检查，以及便于审查的基线和消融流程。其中的[未知蛋白质对挖掘笔记本](../../drafting_code/07_mine_unknown_negatives.ipynb)会审计你导出的 SNOOPPI 未知标签 CSV，使用已有且冻结的 ProtT5 预测器，并可选择将低分、与带标签划分中的蛋白质无身份重叠的**候选**负例保存到 [`mined/`](../guides/unknown-pair-mining.md)。数量以你的 CSV 为准，而非假定固定为 83.5 万条。此流程不训练模型、不确认负例的生物学真实性、不修改带标签的数据划分，也不基于单个检查点提供置信区间。

## 代码所实现的方法

监督学习任务使用 SNOOPPI 的**正例和负例蛋白质对标签**。未知标签的蛋白质对不会加入 ProtT5 的监督训练数据表，也不应视为已确认的负例。ProtT5 主要流程重新加载源数据的正例／负例划分；较早的探索性过滤导出不用于训练。

ProtT5 对蛋白质对的行进行分层随机划分，比例约为 **80% 训练／10% 验证／10% 测试**，划分种子为 **44**。ESM-2 和 ESM-C 复用导出的文件。这种方式不能保证不同划分之间的蛋白质或序列簇互不重叠。ESM-C 会报告原始数据中的重叠，但不会修复；蛋白质对重叠警告不会阻止后续执行。

| 属性 | ProtT5 | ESM-2 | ESM-C（默认） |
| --- | --- | --- | --- |
| 编码器标识 | `Rostlab/prot_t5_xl_half_uniref50-enc` | `facebook/esm2_t33_650M_UR50D` | `esmc_600m` |
| 单个蛋白质表征维度 | 1,024 | 1,280 | 1,152 |
| 对称蛋白质对表征维度 | 3,072 | 3,840 | 3,456 |
| 序列长度限制 | 规范化后的前 1,024 个字符 | 规范化后的前 1,024 个残基 | 规范化后的前 1,024 个残基 |
| 池化 | 均值，排除填充和末尾 EOS | 残基均值，排除 BOS／EOS／填充 | 显式按残基位置取均值，排除 BOS／EOS／填充，保留 `X` |

所有编码器均冻结参数。对于蛋白质表征 `a` 和 `b`，蛋白质对特征由 **`a + b`、`abs(a - b)` 和 `a * b`** 拼接而成。交换两个蛋白质的位置不会改变该表征。

共享分类头的隐藏层宽度为 **512 → 512 → 256**，使用 ReLU、**0.1** 的 dropout 和一个输出 logit。训练批量大小为 **1,024**，优化器为 AdamW，学习率和权重衰减均为 **1e-4**。损失函数为加权二元交叉熵，`pos_weight = n_negative / n_positive`，类别数量取自训练划分。最多训练 **50 轮**，早停耐心值为 **10**。

检查点选择以**验证集宏平均精确率（macro-AP）**最大为准。分类阈值从 0.05 到 0.95 的 181 个候选值中选择，使**验证集 macro-F1** 最大。这些选择表达式不使用测试标签。

## 理解评估指标

- **AUROC** 衡量排序区分能力。
- **正类 AP（Positive AP）** 是标签 1 的平均精确率，其基准水平为正类在数据中的比例。
- **负类 AP（Negative AP）** 对 `1 - label` 和 `1 - score` 进行相同计算。
- **Macro-AP** 是正类 AP 和负类 AP 的算术平均值；它不同于只计算正类的 AP，也不同于用梯形法计算的精确率–召回率曲线下面积。
- **Macro-F1** 是两个类别 F1 分数的平均值，使用在验证集上选定的阈值。

报告这些指标时，应同时提供类别数量和类别占比。尚未证明笔记本中的 sigmoid 分数是经过校准的相互作用概率；它们也不是结合亲和力估计。本次整理没有重新运行实验或验证其输入 CSV，因此不提供数值排行榜。

## 可复现性与局限

本仓库记录已有的实验流程，并未提出新的蛋白质语言模型，也未确立最终的模型排名。笔记本整理保留了所有原始代码单元及其顺序，添加了编号说明，并清除了已保存的输出和临时元数据。已经调试过的实现和历史恢复单元均予以保留。

笔记本说明和[复现说明（英文）](../reference/reproducibility.md)记录了保留的行为，包括序列规范化、检查点恢复和历史单元之间的差异。输入维度不同时，即使隐藏层宽度一致，分类头的总参数量也不同。因此，这些流程并未在完全相同的预处理和模型容量条件下单独比较编码器本身的影响。

[整理清单](../cleanup_manifest.json)记录源文件哈希、按顺序排列的代码单元哈希，以及原始与整理后单元的对应关系。在仓库根目录运行轻量级格式／代码保留检查：

```bash
python -m pip install nbformat
python scripts/validate_notebooks.py
```

此检查验证笔记本结构、代码源文本及顺序未发生变化、已保存输出为空，以及适用单元的 Python 语法。它不下载模型、不挂载 Drive、不训练分类器、不审计数据泄漏，也不验证科学结果。完整运行验证需要 Colab 环境、原始划分文件、模型访问权限和上述执行顺序。本次准备环境中尚未重新进行笔记本渲染和生成图表的视觉验证。

## 引用 SNOOPPI

请引用已发表的期刊论文，保留原始英文题名：

> Vincoff S, Chatterjee P. **SNOOPPI: A Sequence-Normalized Database of On- and Off-Target Protein-Protein Interactions.** *Bioinformatics*. 2026; btag736. [https://doi.org/10.1093/bioinformatics/btag736](https://doi.org/10.1093/bioinformatics/btag736)

<details>
<summary>复制 BibTeX 引用</summary>

```bibtex
@article{vincoff2026snooppi,
  author  = {Vincoff, Sophia and Chatterjee, Pranam},
  title   = {{SNOOPPI}: A Sequence-Normalized Database of On- and Off-Target Protein-Protein Interactions},
  journal = {Bioinformatics},
  year    = {2026},
  pages   = {btag736},
  doi     = {10.1093/bioinformatics/btag736},
  url     = {https://doi.org/10.1093/bioinformatics/btag736}
}
```

</details>

此前的 [OpenReview 研讨会版本](https://openreview.net/forum?id=270Ej8S67w)保留用于追溯来源；当前引用请使用上面的 *Bioinformatics* 论文。

## 上游资源与署名

- [SNOOPPI 期刊论文](https://doi.org/10.1093/bioinformatics/btag736)、[数据集](https://huggingface.co/datasets/ChatterjeeLab/SNOOPPI)和[上游仓库](https://github.com/sophievincoff/snooppi)。
- [ProtT5 编码器](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc)。
- [本仓库使用的 ESM-2 模型](https://huggingface.co/facebook/esm2_t33_650M_UR50D)和[上游 ESM 研究代码](https://github.com/facebookresearch/esm)。
- [ESM-C／EvolutionaryScale ESM](https://github.com/evolutionaryscale/esm)。

请参阅各上游资源的当前引用格式、许可证和模型使用条款，并在论文中记录实际使用的具体版本。本仓库不为第三方数据或权重额外授予许可。这个私有研究工作区尚未选定项目级代码许可证。

Lang-PINN: 从自然语言到物理信息神经网络的多智能体框架

# Lang-PINN

From Language to Physics-Informed Neural Networks via a Multi-Agent Framework

论文被 **ICLR 2026 Workshop on AI with Recursive Self-Improvement** 录用为 Spotlight！

This paper has been accepted as Spotlight at the ICLR 2026 Workshop on AI with Recursive Self-Improvement!

## 📚 论文链接 / Paper Link

[arXiv:2510.05158](https://arxiv.org/pdf/2510.05158)

## 👥 作者 / Authors

Xin He, Liangliang You, Hongduan Tian, Bo Han, Ivor Tsang, Yew-Soon Ong

## 🎯 研究动机 / Motivation

物理信息神经网络（Physics-Informed Neural Networks, PINNs）是求解偏微分方程（PDEs）的强大工具，但传统构建流程繁琐且易出错，严重限制其广泛应用：

* 科学家需手动将实际问题转化为 PDE 形式
* 需精心设计网络架构和损失函数
* 需实现稳定的训练流程，对专业知识要求极高

## 🔧 核心方法 / Key Methods

Lang-PINN 提出了一个 LLM 驱动的多智能体系统，能够从自然语言任务描述自动构建可执行的 PINN 求解器，无需人工干预核心流程。系统由四个协作智能体组成：

### 1. PDE Agent（PDE 智能体）

解析自然语言任务描述，提取偏微分方程中的算子、系数以及边界/初始条件，将其转化为符号化的 PDE 表示，解决“自然语言到数学方程”的转化难题。

### 2. PINN Agent（PINN 智能体）

根据 PDE 的核心特征（周期性、几何复杂度、多尺度动态等），自动选择合适的神经网络架构和归纳偏置，匹配 PDE 特性与网络能力。

生成模块化、可直接执行的 PINN 训练代码实现，采用模块化设计，便于错误定位和组件复用，避免传统单块代码的脆弱性。

### 4. Feedback Agent（反馈智能体）

执行生成的代码，诊断运行时错误，分析训练过程中的残差和收敛性，并向前面的智能体提供迭代式修正反馈，确保最终输出的科学有效性和可训练性。

## 📊 实验结果 / Results

在 14 个不同维度（1D/2D/3D/ND）的 PDE 基准测试中，Lang-PINN 显著优于现有基线方法，核心效果如下：

* ​**误差降低**​：均方误差（MSE）降低了 3\~5 个数量级
* ​**执行成功率**​：端到端执行成功率提升超过 50%
* ​**计算效率**​：时间开销减少高达 74%
* ​**可靠性**​：在 1D 和 2D 场景下成功率超过 80%，而基线方法通常低于 35%

## 🎨 框架流程 / Framework Overview

Lang-PINN 形成闭环式工作流程，从自然语言输入到可运行 PINN 代码输出，全程自动化：

自然语言任务描述 → PDE Agent（符号化 PDE） → PINN Agent（架构选择） → Code Agent（模块化代码） → Feedback Agent（错误诊断与迭代修正） → 最终可执行 PINN 程序

## 📌 关键词 / Keywords

Physics-Informed Neural Networks (PINNs), Large Language Models (LLMs), Multi-Agent System, Partial Differential Equations (PDEs), Scientific Computing

## 📄 相关资源 / Related Resources

* 论文附录：包含详细的实验设置、特征编码细节、数据集说明
* 复现仓库：匿名仓库包含源代码、实验脚本和配置文件（见论文补充材料）

## © 版权说明 / Copyright

Copyright 2026 Xin He. All rights reserved.

Last updated: May 12, 2026

# GAC-Qwen3.5-4B 发布说明

这是 GAC 在 Qwen3.5-4B 紧凑模型骨干上的公开 release。

本次发布重新基于 Qwen3.5-4B 训练 GAC，让自适应 SFT–RL 混合方法跟随小模型的发展，并为研究与部署提供更易使用的 4B 模型。发布包含推理权重、模型配置、评测汇总及公开方法与 evaluator 的链接。

评测表采用作者提供的发布结果。AMC、AIME25、HumanEval 分别更正为 67.5%、20.0%、81.1%；BBH 宏平均按三项展示分数计算为 93.7%。MMLU-Pro evaluator 已固定数据版本及 1,000 题 ID 清单；GPQA 需获得数据访问权限或自行提供合法取得的完整 CSV。

完整英文模型卡、评测表、复现代码和引用信息见 [`README.md`](README.md)。

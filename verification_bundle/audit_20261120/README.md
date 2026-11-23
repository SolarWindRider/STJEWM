# README — 2026-11-22 审计包(v5)

执行:P0-1(oodc_F1 三 seed 动力学,补训 6 ckpt)、P0-2(seed2 异常分解→分母塌缩)、P0-3(λ=0 同日程 control+闭环+bootstrap)、P0-4(剂量匹配掩码)、P0-C(fig4b n=128 修正+v3 面板)、P1-部分(λ 闭环已完成=上轮;随机 floor 参考未做)。

**未完成/边界**:P1-1 动作通路剂量响应未跑(等指令);P1-3 E9 扩样未跑(维持探索性定位);P2 一致性表见 audit.md v2-v5 各节(大部分已修,Training 参数表待补)。

核心文件:audit.md(逐节结论)、各 CSV(§5.1 字段对齐)、figs/(v3 面板+脚本)、scripts 由 export_*.py 与 digest_lambda.py 组成。解释(≤300 字)见 audit.md 末尾。

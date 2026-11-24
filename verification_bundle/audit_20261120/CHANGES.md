# CHANGES(v2,2026-11-20 第二轮)
1. results.csv 重生成(118×14,0 错位);v1 的 22 条错位行与 Fig.2c 问号猜测值删除。
2. fig4b_panel_v2:SLIF obs h5 17.9 占位 → 灰格 n/a †(v1 保留于 20260918/fig3c_panel.*)。
3. fig5b_panel_v2:加 † 图内定义(measurement floor);v1 保留。
4. fig2c_arms.csv:精确值 1.36e-4/34.705/46.335/20.820 + ckpt 对应 + 配方。
5. fd_perenv.csv:SEED_WIN=0、n_win=64 确定性逐环境重导出(pooled 与原表逐位一致);第一版 128-窗导出作废。
6. e1_per_episode.csv(6500 行)与 e9_per_condition.csv 新增。
7. run_plan.md 批次 2 执行语义修正(环境执行完整 25 步序列)。
8. 面板 v2 已同步至 figs_nmi/20261120/(fig4b_panel_v2.*、fig5b_panel_v2.*)。
9. P1 λ 闭环评估完成(3 arms × 100 cells × 5 eps = 1500 episodes + λ=0 复用):配对差 −4.0/−8.6/−5.6 pp(lambda_closedloop_summary.csv、lambda_paired.csv)。
10. P0-B 跨 seed 动力学交付(dynamics_per_window.csv 57600 行、dynamics_summary.csv、dynamics_per_seed_summary.csv、dynamics_models_sr_crossbenchmarkF1.csv);三观察 9/9 复现;trace seed2(cross_benchmark_F1)E_lat 病理标记。
11. P0-C:fig4b_heatmap.csv 以同源逐窗表重导(n=128);新增 fd_perwindow_oodcF1_seed0.csv(10240 行);SLIF obs h5"underflow"定性为 per-env 聚合伪影(cheetah dz(h1)=0 → per-env ratio 2.3e5),直接池化值 1.9e-4 可测。
12. fig4b_panel_v3:全部格子改直接池化增益,SLIF obs h5 = 2e-04 实测,17.9 占位/n/a 退役;已同步 figs_nmi/20261120/。
13. P0-1:oodc_F1 seed1/2 补训 6 ckpt + 同 split 三 seed 动力学(oodc_F1_dynamics_*.csv);seed0 逐位复现已发表值。
14. P0-2:latent_error_decomposition.csv/latent_error_summary.csv;seed2 异常定性=分母塌缩;"预测发散"表述撤回。
15. P0-3:lam00 control 训练+闭环(100 cells)+held-out decoder score;lambda_* 表更新;结论改写为"继续训练解释 SR 下降"。
16. P0-4:maskdose_per_window.csv/maskdose_summary.csv;剂量匹配后 history-dependence 保留(trace B/A 2.2–3.2;gru/lewm C/A≈1)。
17. lambda_bootstrap_ci.csv:cell 级 10k bootstrap CI(含 control)。
18. R1:lambda_vs_continued_control.csv + lambda_vs_control_bootstrap.csv(cell 20k + split-cluster,seed 0);audit.md v5 错误结论撤回改写(仅 λ=1 相对 control 显著下降)。
19. R2:probe_control_comparison.csv + probe_predictions.npz(独立冻结 latent probe);decoder_score_reconciliation.csv(receipt 值为训练尾、非 held-out)。
20. R3:checkpoint_manifest_oodcF1_new.csv(含 SHA256);trace oodc_F1 seed1/2 由 2 层重训为 4 层(首训误用 n-layers 2,已归档);P0-1 数值以 4 层重测为准;lambda_episodes.csv(100/100 配对核验);lambda_arm_checkpoints.csv。
21. R4:maskdose_per_window.csv v2 增输入范数/clean latent norm/绝对位移;cartpole 1 维、cheetah 2 维说明。
22. (2026-11-23 补齐全部未完成实验)随机/零动作参照:零动作 0.477 > 随机 0.356 > Trace 0.311 → "闭环控制能力"绝对主张撤回。
23. 动作通路剂量响应(γ=1/0.75/0.5):无显著效应;sham 揭示同会话确定、跨会话偏移 +0.82 pp(rerun_noise_summary.json)。
24. λ 多训练 seed(8 新 ckpt + 10 × 100 cells):对 control 无跨 seed 一致效应;独立 probe 三 seed 一致。
25. E9 扩样(4 envs × 50 eps):finger ablate_all +40 pp [+24,+56];无一处显著下降。
26. P13 多 epoch、G2 事件 AUROC(676 probes)、utility 4 driver(含新训 6 预算 + 12 G14 ckpt)、P22(零动作操作化)、E4 审计(26 cells 泄漏)全部完成。
27. utility latent_env_grad 修复 cuDNN RNN backward 模式 bug(code/scripts/utility/latent_env_grad.py)。
28. (2026-11-24) 主动运动目标协议 AGR:目标 ≥2×成功容差且零动作 50 步不可达;closed_loop.py 新增 --active-goal-disp/--active-goal-cache。
    13 模型 × 81 cells × 5 eps;零动作地板 0.0000(有效性检查);随机 0.031;全部模型 0.012–0.049 → 无控制能力证据。
29. (2026-11-24) LeWM 核查(audit v9):论文只验证了 4 个像素任务;本机官方栈复现 PushT 88%、TwoRoom 88%;官方权重×官方规划器走我们的环境循环 86–90% → 环境侧正确。
    我们的 LeWM 基线 = 1 epoch/5,000 步/16 环境共享 16 万样本的自写状态版,13 模型单步 latent 预测 R² 全负;忠实状态版 LeWM 训 2 万步即 77%,我们的架构分布内 9 万步仍 0%。
    修复 train.py 在 history_size>1 时的预测目标错位(H=1 不变);发现 CEM 提议分布、DMC 末态判定、DMC 零速度初态三项协议缺陷。全部闭环 env-SR 与 LeWM 相关排序冻结。
    脚本与原始结果:`lewm_check/`。H9 脚本内嵌 env_success_anytime 读数作废(目标未除 500),以 h10 修正评测为准。

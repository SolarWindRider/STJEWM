# 协议(protocol.md)

全部从代码核对(2026-10-02 协议审计,含一次漏加载 bug 的修复记录)。写作时 Methods 描述以此为准。

## 1. 训练协议(简要)

- 入口:`code/train/train.py`(单 cell 驱动:`code/scripts/generalist_v0_7_5_5m/train_one_5m.sh`);批量重训:`repair_training.py --manifest .../training_final_repair_manifest.json --execute`(371/371)。
- 损失:**三段式 JEPA**(非重建):`pred MSE + sigreg + goal`(`code/train/native_losses.py::stjewm_loss`;预测头在表示空间预测未来 latent,stop-grad 目标)。
- 5M 对齐:**口径 = 可训练参数**(排除 STJEWM 冻结的 ImageNet ViT-Tiny 编码器)。实测(2026-10-21,按 ckpt 逐张量统计):state trace 总 10.523M = 冻结 encoder 5.459M + 可训练 5.064M;pixel trace 总 10.461M / 冻结 5.459M / 可训练 5.002M;基线全可训练 lewm 4.970M、gru 5.132M、mlp 5.001M → 全部落在 4.97–5.13M(±3.2%)。**论文的 "5.06M trainable" 与此一致;写作时必须声明 STJEWM 额外携带 5.46M 冻结编码器,不计入匹配预算。**
- 数据:DMC 250k 离线轨迹(npz,低维本体感知态);pusht/tworoom 用 LeWM h5(绝对路径 `/home/lx/LeWM/data/…`);窗口 53 帧,目标位动作零填充。
- G 规模轴:嵌套 union(G4⊂G8⊂G16),env 数 4/8/16,每 env 窗口 2000/2000/10000(**规模轴同时混杂两个因子,报告 §12.9 有明确警示,写作须保留**)。

## 2. State 闭环评测(冻结协议,E1/E2/E4/E7/E13/E14 全部同协议)

入口 `python -m code.eval.closed_loop`(严格加载,打印 `trained weights loaded strictly`);常数与环境变量见 `code/scripts/generalist_v0_7_5_5m/eval_one.sh`。

### 2.1 从 o_t 到动作(逐步)

1. **状态准备** `model_state()`:原生状态 → 观测掩码(如 qpos-masked)→ /state_scale(pusht 500、tworoom 250,DMC 1.0)→ 零填充至 128。
2. **上下文编码** `encode_obs`(history_size=1):`s=o_t (1,1,128)`,`a=0 (1,1,56)` → 完整前向 `h_in=E_o+E_a(0)` → SNN stack 1 步 → readout(trace_only: `z=trace_proj(R)`)→ `z_t ∈ R^192`。
   - a≡0 的依据:规划时当前动作不存在;训练约定窗口末位动作本为零填充;**2026-10-02 消融证据**(imag_position_sensitivity.json + protocol_ablation.py):上下文动作通道对闭环无影响(信息由观测通道携带),规划依赖的动作敏感性在 predict 通路。
3. **目标编码**(每 episode 一次):`z_goal = encode_obs(o_goal)`,同一函数、同一零动作约定。
4. **CEM**(`cem.py::plan`):μ=0,σ=1;10 iters × 300 候选 × H=25;每候选从 z_t 起 25 步 `predict` 想象(候选动作 clamp→pad 56 后**直接落在读出位**),代价 = ‖z_final − z_goal‖²;精英 30;返回 argmin 序列。
5. **执行**:**整条 25 步开环执行**(非 first-action MPC),budget 50 步 → 每 episode 恰好 2 次 planning calls;之后新 o_t 重规划(丢弃旧 latent,无跨规划记忆)。
6. **度量**:`z_final = encode_obs(o_final)`;`cos_dist=(1−cos)/2`;env-SR 见 §2.3。

### 2.2 Goal 构造(回答审稿人 Methods §Goal construction 质疑)

- goal = **同一真实训练轨迹**中 t+25 的**记录状态**(非模型生成、非随机状态);`_load_eval_dataset(split="in_dist")` 从离线 npz 取 `(init_state, goal_state)` 对。
- 初始化:恢复记录的原生初始态;"未记录的速度置零"(协议 JSON `initialization` 字段原文)。
- 与常规 MPC 的区别(写作时须明确声明):goal-conditioned、25 步开环段 × 2 段、goal 固定不可达即失败;**不是**逐步重规划的轨迹跟踪。
- 方法论后果:该协议测"潜状态能否把当前态引导到 25 步外的真实未来态",不测任意目标泛化。

### 2.3 env-SR 判定(回答审稿人问题 2 的精确答案)

**env-SR 用的是真实执行后环境到达的状态,不是模型预测状态。**
代码链:`closed_loop.py` 循环内 `obs, reward, done, info = env.step(action)`(真实环境执行)→ 终局 `final_state = physical_state()`(原生、未掩码)→ `physical_env.check_success(final_state, goal_state)`(环境自带判定,如 DMC `‖Δqpos‖/√nq ≤ 0.1`、pusht 位置 20px+角度 π/9)。
因此:env-SR 是严格的 environment success;`phys_dist` 同样是真实物理距离。Methods 草稿中 `x_pred` 记号是**错误记号**,应写 `x_final`(执行后真实状态)。
另:`cos_dist`/`LeWM-SR` 才是模型侧量(基于 z_final),两者不可混用。

### 2.4 Pixel 闭环(E3)差异

- 观测:DMC 实时像素 → 冻结 ViT(`FrozenPixelPreprocessor`,84px,257 tokens)编码;无 npz。
- **horizon 5、每步重规划**(50 步 = 10 次规划);其余(CEM 300/30、goal offset、度量)同 state。
- 驱动:`eval_pixel_ckpt.py` / `run_pixel_eval_all.sh`。

### 2.5 常见误用警示

- **不要**把 27 帧上下文协议(S 系列机制探针用)写成闭环评测协议——闭环是单帧上下文。
- **不要**用旧代际常数(如 H5 B50 的旧叙述、"CEM 30 iters 全环境"等);以本文件与 `eval_one.sh` 为准。
- cos_dist 可被坍缩 latent 平凡取得低值(λ=0 塌缩模型 cos 0.005);**任何 cos 数字必须伴随 env-SR 或 regime 标注**。

## 2.6 E9 消融的精确干预语义(回答审稿 P0.4)

`event_window_ablation.py:139-150`:两类 hook,都只作用于**规划期 imagination 的 `predict` 调用**,不触碰 CEM 初始编码(z_history/z_goal 的 forward 路径):

- `ablate_all` = `step_hook(all_steps)`:在每个已执行步之后的那次规划里,用 `zero_stjewm_trace(model)` 包裹 `model.predict`——即**把 imagination 期间的 trace 通路置零**;
- `cem_rollout_ablation` = `cem_hook`:仅在 CEM 的候选 rollout 内置零 trace(作用域更窄);
- baseline/event/non_event/random = 仅在特定规划步置零(窗口调度)。

因此 E9 的正确表述:**"规划期想象中的 trace 内容不是 env-SR 的必要条件"(两环境、6 模式、cartpole SR 0.60→0.70)**。它**不**触及:(a) CEM 初始化 latent z_t 的编码(其 trace 读出未被消融);(b) 训练;(c) 其他环境。禁止写成"trace 对成功无因果作用"。

## 2.7 评测数据独立性(回答审稿 P0.6)

- E1/E2 in_dist:eval 的 (init, goal) 对来自**训练所用的同一批 250k 离线轨迹** → 轨迹层面存在训练-评测重叠;成功率必须写作 **"dataset-derived, goal-conditioned, physical-environment success"**,不得写成开放式控制/任意目标泛化。
- E4 `heldout_*`(`configs/heldout_eval/`)是**环境组合层面的 held-out**(模型训练 split 的补集环境),**不是**轨迹级 held-out。
- 规模轴 E5/E14 的 union 嵌套与数据量混杂见 §1 G 规模轴注。

## 2.8 "Event history" 的准确语义(回答审稿 P0.2)

- trace 的"事件历史"是**单次前向内部**的 SNN 时间整合:训练窗口 53 帧;评测期想象 rollout 25 步(`predict` 自回归过 recurrent core)。
- 真实环境时间上,**规划边界无记忆**(history_size=1:每次重规划只编码最新观测;前次 plan 的 latent 丢弃)。
- 所以:标题/示意不得暗示 agent 跨真实时间维持 event-history 状态;应写 "within-rollout event integration"(训练与想象 rollout 内部),在线状态估计 = 最新单帧观测的读出。

## 2.9 Pixel vs State 不是单因素对照(回答审稿 P0.7)

state H=25×2 段 vs pixel H=5×10 段:跨模态差异同时受视觉输入与控制协议影响。跨模态陈述限定为"协议内重复观察",不得写成受控对比;若需单因素结论须补同协议对照(**未做**)。


全部经 `mech0917_multistep.load_model`(严格加载)。窗口 27 帧、pad 128/act 56。关键常数:

| 探针 | 设计 | 脚本 |
|---|---|---|
| S3 gain | 配对 ±0.1σ 脉冲于窗口末位/全窗,测 g_obs、g_a | `stab0917_gain.py` |
| S4 obs 脉冲 | obs 脉冲响应衰减(1 步内归零) | `stab0917_maskobs.py` |
| M4 瞬态恢复 | 高斯 obs 噪声(1σ)注入 m∈{1,2,4,8} 步,T_recover = 回到 0.1·d_peak;含持续扰动对照 | `mech0917_transient.py`(2026-10-02 修复 strict-load 后重测,**旧版数字作废**) |
| 位置敏感性 | 单动作于窗口位置 p,测 predict 末位相对响应 | `imag_position_sensitivity.py`(2026-10-02 新增) |

位置敏感性结论(真权重):trace 全位置响应 0.23–1.27;LeWM 仅末位 0.018、其余 1.5e-4 → **官方闭环协议(history_size=1)不存在"失明"问题**(候选动作永远在唯一读出位);该诊断只用于解释 S 系列 27 窗口协议下的现象。

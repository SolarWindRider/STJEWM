# 协议(protocol.md)

全部从代码核对(2026-10-02 协议审计,含一次漏加载 bug 的修复记录)。写作时 Methods 描述以此为准。

## 1. 训练协议(简要)

- 入口:`code/train/train.py`(单 cell 驱动:`code/scripts/generalist_v0_7_5_5m/train_one_5m.sh`);批量重训:`repair_training.py --manifest .../training_final_repair_manifest.json --execute`(371/371)。
- 损失:**三段式 JEPA**(非重建):`pred MSE + sigreg + goal`(`code/train/native_losses.py::stjewm_loss`;预测头在表示空间预测未来 latent,stop-grad 目标)。
- 5M 对齐:全模型 4.97–5.13M 参数(±3.2%);STJEWM 总 16.57M / 可训练 10.54M(encoder 冻结 ViT 部分不计入 5M 口径,见 README §4.12——**写作时参数口径必须注明**)。
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

## 3. 机制探针协议(S/M 系列,27 帧上下文)

全部经 `mech0917_multistep.load_model`(严格加载)。窗口 27 帧、pad 128/act 56。关键常数:

| 探针 | 设计 | 脚本 |
|---|---|---|
| S3 gain | 配对 ±0.1σ 脉冲于窗口末位/全窗,测 g_obs、g_a | `stab0917_gain.py` |
| S4 obs 脉冲 | obs 脉冲响应衰减(1 步内归零) | `stab0917_maskobs.py` |
| M4 瞬态恢复 | 高斯 obs 噪声(1σ)注入 m∈{1,2,4,8} 步,T_recover = 回到 0.1·d_peak;含持续扰动对照 | `mech0917_transient.py`(2026-10-02 修复 strict-load 后重测,**旧版数字作废**) |
| 位置敏感性 | 单动作于窗口位置 p,测 predict 末位相对响应 | `imag_position_sensitivity.py`(2026-10-02 新增) |

位置敏感性结论(真权重):trace 全位置响应 0.23–1.27;LeWM 仅末位 0.018、其余 1.5e-4 → **官方闭环协议(history_size=1)不存在"失明"问题**(候选动作永远在唯一读出位);该诊断只用于解释 S 系列 27 窗口协议下的现象。

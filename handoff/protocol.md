# 协议(protocol.md)

全部从代码核对(2026-10-02 协议审计,含一次漏加载 bug 的修复记录)。写作时 Methods 描述以此为准。

## 1. 训练协议(简要)

- 入口:`code/train/train.py`(单 cell 驱动:`code/scripts/generalist_v0_7_5_5m/train_one_5m.sh`);批量重训:`repair_training.py --manifest .../training_final_repair_manifest.json --execute`(371/371)。
- 损失:**三段式 JEPA**(非重建):`pred MSE + sigreg + goal`(`code/train/native_losses.py::stjewm_loss`;预测头在表示空间预测未来 latent,stop-grad 目标)。
- 5M 对齐:**口径 = 可训练参数**(排除 STJEWM 冻结的 ImageNet ViT-Tiny 编码器)。实测(2026-10-21,按 ckpt 逐张量统计):state trace 总 10.523M = 冻结 encoder 5.459M + 可训练 5.064M;pixel trace 总 10.461M / 冻结 5.459M / 可训练 5.002M;基线全可训练 lewm 4.970M、gru 5.132M、mlp 5.001M → 全部落在 4.97–5.13M(±3.2%)。**论文的 "5.06M trainable" 与此一致;写作时必须声明 STJEWM 额外携带 5.46M 冻结编码器,不计入匹配预算。**
- 深度:**STJEWM 的 spiking stack 为 4 层**(`n_layers=4`,stack 4.732M;已发表 seed0 与 G5/B2 的 seed1/2 ckpt 均为 4 层,2026-11-23 逐 ckpt 核对)。⚠️ `G5/B2_multiseed_launcher.sh` 文本中的 `--n-layers 2` 对 stjewm 具误导性(实际 ckpt 是修复重训的 4 层代际);**任何新训 trace 必须 `--n-layers 4`**——2026-11-23 曾因照抄该参数训出 2 层 oodc_F1 seed1/2(8.16M),已作废并以 4 层重训。
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
5. **执行**:**整条 25 步开环执行**(非 first-action MPC;pixel 侧语义不同,见 §2.4),budget 50 步 → 每 episode 恰好 2 次 planning calls;之后新 o_t 重规划(丢弃旧 latent,无跨规划记忆)。代码证据:`closed_loop.py:277` `for candidate in seq[:min(horizon, ...)]` 整段逐步执行后才回 while 重规划。
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
- **horizon 5、每步重规划 = first-action MPC**(CEM 只优化 H5 序列但**只执行首个动作**,每步拿新观测重规划;50 步 = **50 次** planning calls;`eval_pixel_ckpt.py:5` "replanning after every executed action"、`:219` `action_seq[0]`);其余(CEM 300/30、goal offset、度量)同 state。
- ⚠️ **执行语义红线(2026-10-22 代码核实)**:state 与 pixel **不同**——state=分段开环(§2.1,pooling 后每 episode 2 次 planning calls),pixel=逐步 first-action MPC(50 次)。Methods **不得**写成统一的 "receding-horizon MPC";必须分 modality 表述。E9 trace-ablation(state,H5)为分段执行,10 段/episode(`eligible_planning_steps=0,5,…,45`)。
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


## 3. 机制探针协议(27 帧窗口)

全部经 `mech0917_multistep.load_model`(严格加载)。窗口 27 帧、pad 128/act 56。关键常数:

| 探针 | 设计 | 脚本 |
|---|---|---|
| S3 gain | 配对 ±0.1σ 脉冲于窗口末位/全窗,测 g_obs、g_a | `stab0917_gain.py` |
| S4 obs 脉冲 | obs 脉冲响应衰减(1 步内归零) | `stab0917_maskobs.py` |
| M4 瞬态恢复 | 高斯 obs 噪声(1σ)注入 m∈{1,2,4,8} 步,T_recover = 回到 0.1·d_peak;含持续扰动对照 | `mech0917_transient.py`(2026-10-02 修复 strict-load 后重测,**旧版数字作废**) |
| 位置敏感性 | 单动作于窗口位置 p,测 predict 末位相对响应 | `imag_position_sensitivity.py`(2026-10-02 新增) |

位置敏感性结论(真权重):trace 全位置响应 0.23–1.27;LeWM 仅末位 0.018、其余 1.5e-4 → **官方闭环协议(history_size=1)不存在"失明"问题**(候选动作永远在唯一读出位);该诊断只用于解释 S 系列 27 窗口协议下的现象。

## 4. 诊断协议常数(2026-11-20~23 审计核定,全部从脚本核对)

| 协议 | 窗口 | 扰动 | 量 | 脚本 |
|---|---|---|---|---|
| S1 rollout 稳定性(Fig.3b/3c/5a 的 c20) | 128 窗/env,`default_rng(0)`,27 帧真实窗 | 末位上下文 latent `z_t += ε·u`,ε=0.05×mean‖z*‖(warm 全 episode),u 单位随机方向;paired twin rollout,教师强制 recorded actions,H=20 | c(h)=mean‖ẑ_pert−ẑ‖/ε;c20=c(20) | `stab0917_rollout.py` |
| FD 增益(Fig.4a/4b/5b) | **64 窗/env × 2 env = 128 窗**(`SEED_WIN=0`) | paired ±0.1×per-dim std,作用于**末位上下文帧**;action 改 a[-1],obs 改 o[-1] 后**重编码整窗**;后续用 recorded actions 做 latent rollout,H=5 | g(h)=mean‖z₊−z₋‖/(‖z_base(h)‖+1e-8) | `cls0917_scaling.py` |
| CSR | 同上 | — | **CSR_env = mean_e[g_a,e/(g_o,e+1e-8)]**(逐 env 取比值再平均,Fig.5b 口径);**CSR_pooled = mean(g_a)/(mean(g_o)+1e-8)**(Fig.4a 坐标比值) | 同上 |
| Fig.4b h=5 格 | 同上 | — | **直接池化 g(5)**(v3 面板);v1 用 g(1)×mean-dz(5)/mean-dz(1),对 SLIF obs h5 产生 per-env 聚合伪影 | `make_fig4b_v3.py` |
| dose c20(Fig.2c) | 32 窗/env(128 窗取样的前 32) | `ctx[:,-1] += ε·u`,ε=0.05×**std(z*)** | c20=disp(20)/ε | `access_eval.py` |
| 遮蔽位移(Fig.4c) | 128 窗/env × 3 mask seed | 置零(非噪声):last25=末帧 0.25D 维;win25=27 帧每帧同 25% 维(27× 剂量) | rel_l2=‖Δz‖/(‖z_clean‖+1e-8) | `stab0917_maskobs.py` |
| 剂量匹配遮蔽(P0-4) | 64 窗/env × 3 mask seed | A last25(末帧 0.25D 个元素)vs B scatter25(**同总元素数**随机散布 27 帧)vs C win25(27× 参考);0.25D 四舍五入且≥1 → cartpole 1 维、cheetah 2 维 | 同上 + 输入 ΔL2、绝对位移 | `export_maskdose.py`(审计包) |

**跨模型尺度警示**:trace 的 clean latent norm ≈0.01,比 GRU/LeWM/membrane(12–32)小约 1000×——相对位移比由小分母放大,跨模型比较须用绝对量或注明尺度。
**口径警示**:S1 的 E_lat(1) 是 teacher-forced latent rollout 相对 L2(非开环单步预测误差);其分母 ‖z*‖ 在个别 ckpt 上可塌缩(trace seed2@cross_benchmark_F1:21–70% 窗口 <1e-3),此时该指标不可解释(errata.md)。

## 5. 参照策略与可复现性(2026-11-23)

- **零动作 / 随机动作参照**:`code/scripts/extra_closed_loop_evals.py --mode zero|random`——在 `eval_closed_loop` 内把 `CEM.plan` 替换为零序列 / 均匀随机(native box),其余(实例 manifest、init/goal、50 步预算、2 次规划、终点判据)与主协议完全相同。写作时任何 env-SR 都必须与这两条地板并列。
- **动作通路剂量响应**:`--mode dose --gamma γ`,经 `cem_predict_hook` 在候选评分内把动作改为 a′ = γ·a(参考动作 ā=0),环境执行原始序列;γ=1 为 sham。
- **可复现性**:同一会话(同 GPU/库环境)内闭环评测逐 episode 确定;跨会话存在偏移(9 月 E1 vs 11 月:agreement 0.909,+0.82 pp)。**跨会话的配对比较须用同会话基线**。
- **E4 held-out 单位更正**:546 cells 中 520 为未见环境;26 cells(oodc_F3/oodc_F2F3 训练的模型评 cheetah)为 spec 泄漏,须排除。
- **新训 ckpt 的配方**:trace 一律 `--n-layers 4`;预算实验用 `max_windows` 5000/20000 的 G16 spec;G14 用 `configs/generalist_G16_minus_walker_humanoid.json`。这些 ckpt 不在冻结 training manifest 内(post-freeze),只能经独立评估使用,不得走 manifest 门控的官方 driver。

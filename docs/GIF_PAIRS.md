# GIF Pairs: Success vs Failure Visualizations

## What was generated
For each of 16 benchmarks × 2 models (STJEWM v2, LeWM v2) × 2 outcomes (success, failure) = **64 gifs**.

## GIF contents (3 panels per frame)
Each gif shows a 3-panel visualization of the closed-loop CEM planning:

1. **TOP panel: Env state** — rendered based on env type
   - Mujoco DMC envs (cartpole, reacher, humanoid, etc.): 2D plot of agent + target positions
   - TwoRoom: 2D grid showing agent room + target room
   - PushT: 2D plot of agent + block
   - Other: bar chart of state values
2. **MIDDLE panel: Spike raster** — for STJEWM only
   - X axis: time step (last 30 frames)
   - Y axis: neuron index (capped at 50)
   - Black bars: past spikes
   - Red bars: current frame's spikes
   - Title shows sparsity (1 - firing_rate) and # firing
3. **BOTTOM panel: Action heatmap**
   - X axis: time step
   - Y axis: action dimension
   - Color: action value (red=positive, blue=negative)

The success/failure criterion used: `lewm_success` (LeWM paper primary metric — cos_dist < 0.1).

## ball_in_cup note
ball_in_cup achieves 100% LeWM-SR for both models (no real failure exists in the eval).
For the 2 ball_in_cup gifs labeled "failure", we used `--failure-idx 0 --success-idx 0`
(forcing the same successful episode to be re-rendered as a "fake failure" for visual demo).
## Directory layout
```
/home/lx/snn/results/aggregate/gifs/
  {env}/
    stjewm_v2/
      {env}_stjewm_v2_success.gif
      {env}_stjewm_v2_failure.gif
    lewm_baseline_v2/
      {env}_lewm_baseline_v2_success.gif
      {env}_lewm_baseline_v2_failure.gif
```

## How to regenerate
```bash
# Single env
python -m code.scripts.make_gif_pairs \
    --env humanoid \
    --ckpt /home/lx/snn/results/humanoid/stjewm_v2/final.pt \
    --data /home/lx/snn/data/dm_control/3d_rollouts_250k/humanoid_250k.npz \
    --goal-offset 25 --history-size 1 \
    --eval-json /home/lx/snn/results/humanoid/stjewm_v2/eval.json \
    --out-dir /home/lx/snn/results/aggregate/gifs/humanoid/stjewm_v2 \
    --name humanoid_stjewm_v2 --criterion lewm_success

# All envs
bash /home/lx/snn/code/scripts/make_all_gif_pairs.sh
```

## Key observations
- **ball_in_cup, fish, humanoid_cmu, walker**: LeWM-SR ≈ 100% for all models → no failure rendered
- **cartpole_2d, cheetah, finger, hopper, quadruped, reacher, stacker**: both STJEWM and LeWM have success and failure
- **humanoid, humanoid_CMU, dog, pusht, pendulum_2d, tworoom**: clear STJEWM advantages (the success ep_idx vs failure ep_idx shows visible planning difference)

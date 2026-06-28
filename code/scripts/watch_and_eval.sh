#!/bin/bash
# Watch for STJEWM phase completion, then auto-launch eval.
set -e
cd /home/lx/snn
echo "Waiting for STJEWM phase to complete..."
while true; do
    GPU0_DONE=$(grep -c "ALL TRAININGS COMPLETE" /tmp/stjewm_train_logs_gpu0/all.log 2>/dev/null || echo 0)
    GPU1_DONE=$(grep -c "ALL TRAININGS COMPLETE" /tmp/stjewm_train_logs_gpu1/all.log 2>/dev/null || echo 0)
    if [ "$GPU0_DONE" -ge 1 ] && [ "$GPU1_DONE" -ge 1 ]; then
        break
    fi
    RUNNING=$(ps aux | grep -c "code.train.train" | grep -v grep || echo 0)
    if [ "$RUNNING" -eq 0 ] && [ "$GPU0_DONE" -ge 1 ] && [ "$GPU1_DONE" -ge 1 ]; then
        break
    fi
    sleep 60
done
echo ""
echo "STJEWM phase complete. Launching eval phase..."

mkdir -p /tmp/eval_logs_stjewm /tmp/eval_logs_lewm_baseline

# Split envs between 2 GPUs
# GPU 0: 8 envs, GPU 1: 7 envs
GPU0_ENVS="cheetah walker hopper quadruped humanoid humanoid_CMU dog fish"
GPU1_ENVS="stacker reacher finger ball_in_cup cartpole_2d pendulum_2d pusht tworoom"

for gpu in 0 1; do
    if [ "$gpu" -eq 0 ]; then
        ENVS=$GPU0_ENVS
    else
        ENVS=$GPU1_ENVS
    fi
    for model in stjewm lewm_baseline; do
        CUDA_VISIBLE_DEVICES=$gpu setsid nohup bash -c "MODEL_FILTER=$model /home/lx/snn/code/scripts/eval_all.sh $ENVS" \
            > "/tmp/eval_logs_${model}/gpu${gpu}.log" 2>&1 &
        disown
    done
done

echo "Eval launched. Tails: /tmp/eval_logs_stjewm/* and /tmp/eval_logs_lewm_baseline/*"

#!/bin/bash
# Super watchdog: aggressive relaunch + monitoring. 
# All 3 pipelines stay alive at all times.
set +e
cd /home/lx/snn

while true; do
  cur_train=$(find results/5m/ -name "final.pt" 2>/dev/null | wc -l)
  cur_eval=$(find results/5m/ -name "eval_*.json" 2>/dev/null | wc -l)
  cur_probe_ok=$(find results/probe_5m/ -name "*.json" -exec grep -l '"skipped": false' {} \; 2>/dev/null | wc -l)
  
  # Check what's running
  active_train=$(ps aux | grep -E "python.*train\.train" | grep -v grep | wc -l)
  active_eval=$(ps aux | grep -E "closed_loop" | grep -v grep | wc -l)
  active_probe=$(ps aux | grep -E "code.scripts.probe " | grep -v grep | wc -l)
  
  echo "[super $(date +%H:%M:%S)] train=$cur_train/130 (active=$active_train) eval=$cur_eval (active=$active_eval) probe=$cur_probe_ok (active=$active_probe)"
  
  if [[ $cur_train -ge 130 ]] && [[ $cur_eval -gt 600 ]] && [[ $cur_probe_ok -gt 1500 ]]; then
    echo "[super] all done at $(date -Iseconds)"
    break
  fi
  
  # Launch any pipeline that's idle
  if [[ $active_train -eq 0 ]] && [[ $cur_train -lt 130 ]]; then
    echo "[super] launching train"
    nohup bash code/scripts/generalist_v0_7_5_5m/launch_parallel.sh > /tmp/launch/train_$(date +%H%M%S).log 2>&1 &
    disown
    sleep 5
  fi
  
  if [[ $active_eval -eq 0 ]] && [[ $cur_train -gt 0 ]]; then
    echo "[super] launching eval"
    nohup bash code/scripts/generalist_v0_7_5_5m/eval_all.sh > /tmp/launch/eval_$(date +%H%M%S).log 2>&1 &
    disown
    sleep 5
  fi
  
  if [[ $active_probe -eq 0 ]] && [[ $cur_train -gt 0 ]]; then
    echo "[super] launching probe"
    nohup bash code/scripts/generalist_v0_7_5_5m/probe_all.sh > /tmp/launch/probe_$(date +%H%M%S).log 2>&1 &
    disown
    sleep 5
  fi
  
  sleep 30
done

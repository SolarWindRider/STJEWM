#!/bin/bash
# Evaluate all twelve external checkpoints into a fresh, complete diagnostic grid.
set -euo pipefail
cd /home/lx/snn
export MUJOCO_GL=egl
export PYTHONPATH=/home/lx/snn
PY=${PY:-/home/lx/miniconda3/envs/snn/bin/python}
"$PY" -u - <<'PY'
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from code.scripts.eval_spiking_wm_protocol import DMC_TASK_MAP, sha256

root = Path(os.environ.get('CKPT_ROOT', 'results/spiking_wm')).resolve()
out = Path(os.environ.get('EVAL_ROOT', '/data/lx/tmp/results/spiking_wm_final_20260916')).resolve()
gpus = os.environ.get('GPUS', '0,1,2,3').split(',')
workers = int(os.environ.get('WORKERS', len(gpus)))
if workers < 1 or not gpus or any(not gpu.isdigit() for gpu in gpus):
    raise ValueError('Require positive workers and explicit physical GPU indices')
if out.exists():
    raise FileExistsError(f'Archive existing external diagnostic grid first: {out}')
checkpoints = {task: root / ('logs_' + task) / 'latest_model.pt' for task in DMC_TASK_MAP}
missing = [str(path) for path in checkpoints.values() if not path.is_file()]
if missing:
    raise FileNotFoundError(missing)
out.mkdir(parents=True)
print(f'[external-grid] START tasks={len(checkpoints)} workers={workers}', flush=True)

def run(index, task):
    checkpoint = checkpoints[task]
    destination = out / (task + '.json')
    before = sha256(checkpoint)
    gpu = gpus[index % len(gpus)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MUJOCO_EGL_DEVICE_ID=gpu)
    command = [sys.executable, 'code/scripts/eval_spiking_wm_protocol.py',
               '--task', task, '--ckpt', str(checkpoint), '--out', str(destination),
               '--n-steps', '2000', '--seed', '0', '--device', 'cuda:0']
    receipt = dict(task=task, checkpoint=str(checkpoint), checkpoint_sha256=before,
                   output=str(destination), command=command)
    try:
        with (out / (task + '.log')).open('x') as log:
            process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        receipt['returncode'] = process.returncode
        if process.returncode:
            raise RuntimeError(f'External diagnostic exited {process.returncode}')
        result = json.loads(destination.read_text())
        if (result['protocol_version'] != 2 or result['status'] != 'completed'
                or result['weights_loaded_strict'] is not True or result['task'] != task
                or result['n_steps'] != 2000 or result['seed'] != 0
                or result['checkpoint_sha256'] != before or sha256(checkpoint) != before
                or result['raw_arrays_sha256'] != sha256(result['raw_arrays'])):
            raise ValueError('External result failed protocol/provenance admission')
        receipt.update(status='completed', summary_sha256=sha256(destination))
        print('[external-grid] COMPLETE', task, flush=True)
    except Exception as error:
        receipt.update(status='failed', error=str(error))
        print('[external-grid] FAILED', task, str(error), flush=True)
    return receipt

with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [pool.submit(run, index, task) for index, task in enumerate(DMC_TASK_MAP)]
    outcomes = [future.result() for future in as_completed(futures)]
status = {'protocol_version': 2, 'planned_tasks': list(DMC_TASK_MAP), 'outcomes': outcomes}
with (out / 'grid_status.json').open('x') as handle:
    json.dump(status, handle, indent=2, allow_nan=False)
failures = [item for item in outcomes if item['status'] != 'completed']
print(f'[external-grid] completed={len(outcomes) - len(failures)} failed={len(failures)}', flush=True)
if failures:
    raise SystemExit(1)
PY

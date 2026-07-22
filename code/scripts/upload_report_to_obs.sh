#!/usr/bin/env bash
# Upload the v0.7.13 final reports (paper PDF + experiment_report_full_zh.{tex,pdf})
# + the per-env dataset sample figures to OBS.
#
# Layout (under obs://lixiang01/STJEWM_NMI/):
#   paper/paper.pdf
#   experiment_report_full_zh.pdf
#   experiment_report_full_zh.tex
#   paper/figs/fig1-5_*.png          (the 5 main paper figures)
#   paper/figs/dataset_samples/<env>.png        (16 per-env)
#   paper/figs/dataset_samples/all_envs_overview.png
#   paper/figs/dataset_samples/obs_samples.json
#   paper/figs/dataset_samples/generate_samples.py
#
# This script is idempotent (cp overwrites existing keys by default).
# Use -flat to avoid creating a nested directory.

set -e
cd /home/lx/snn
OBSUTIL=/home/lx/obsutil_linux_amd64_5.8.3/obsutil
BUCKET=obs://lixiang01/STJEWM_NMI

# 1) The English paper PDF
echo "== paper/paper.pdf =="
$OBSUTIL cp paper/paper.pdf "$BUCKET/paper/paper.pdf" 2>&1 | tail -1

# 2) The Chinese report PDF + LaTeX
echo "== experiment_report_full_zh.pdf =="
$OBSUTIL cp paper/experiment_report_full_zh.pdf "$BUCKET/experiment_report_full_zh.pdf" 2>&1 | tail -1
echo "== experiment_report_full_zh.tex =="
$OBSUTIL cp paper/experiment_report_full_zh.tex "$BUCKET/experiment_report_full_zh.tex" 2>&1 | tail -1

# 3) The 5 main paper figures (fig1-5)
for f in fig1_protocol.png fig2_scatter.png fig3_specialist_heatmap.png fig4_diagnostic_3panel.png fig5_event_align_ts.png; do
    echo "== paper/figs/$f =="
    $OBSUTIL cp paper/figs/$f "$BUCKET/paper/figs/$f" 2>&1 | tail -1
done

# 4) The per-env dataset sample figures (16 PNGs + overview + JSON + script)
# Use -flat to avoid nested directory, and exclude __pycache__.
echo "== paper/figs/dataset_samples/ (flat) =="
$OBSUTIL cp paper/figs/dataset_samples/ "$BUCKET/paper/figs/dataset_samples/" \
    -r -f -flat -exclude=__pycache__ 2>&1 | tail -3

echo
echo "Done. To verify:"
echo "  $OBSUTIL ls $BUCKET/paper/ | head"
echo "  $OBSUTIL ls $BUCKET/paper/figs/dataset_samples/ | head"

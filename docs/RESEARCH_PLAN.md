# Research Plan v4 — v5 final

## Current state

### Achieved
- **25 3D envs** trained and evaluated (12 dm_control + 4 AdroitHand + 4 Hand + Jaco + Franka + TestArm + UR5e + Robotiq)
- **17 3D arm/hand envs** specifically (all 3D mechanical arm/hand/manipulation benchmarks covered)
- **v5 SNN world model** (5.03M params) achieves:
  - Next-step cos 0.971-1.000 (mean 0.986) on all 25 envs
  - Closed-loop cos 0.975-1.000 (mean 0.987) on 24/25 envs
- **Target-conditioned planning** (manipulator 17D): 87.5% improved
- **v5 SNN beats LeWM Transformer baseline** by +12.5pp on planning
- **Three theoretical propositions** validated by Monte Carlo (N=10,000)

### Open directions
1. **More data + training** (2M+ rollouts, 10+ epochs) for higher reach task success rate
2. **Architectural variants**:
   - Add attention mechanism to MultiComp SNN
   - Deeper model (8+ layers)
   - Larger embed_dim (256+)
3. **Algorithm improvements**:
   - Different CEM (e.g. CMA-ES, MPPI)
   - Model-predictive path integral (MPPI)
   - Hierarchical planning (sub-goals)
4. **Pixel-based extension**:
   - Train v5 on pixel observations (replace state_projector with ViT)
   - Apply to OGBench Cube (image-based 3D arm)
5. **Transfer learning**:
   - Pre-train on one env, fine-tune on another
   - Test few-shot adaptation to new 3D arm
6. **Hardware deployment**:
   - Test on Loihi (Intel neuromorphic chip)
   - Energy benchmarking vs GPU
7. **Theory extension**:
   - More propositions (planning convergence, sample efficiency)
   - Tighter bounds on SIGReg regularization

## Bench coverage status

| Source | 3D arm/hand envs | Status |
|--------|------------------|--------|
| dm_control | 4 (manipulator, stacker, finger, ball_in_cup) | DONE |
| dm_control | 8 (locomotion, not arm) | DONE (extra) |
| AdroitHand | 4 (door, hammer, pen, relocate) | DONE |
| Hand | 4 (reach, block, egg, pen) | DONE |
| Jaco | 1 (arm) | DONE |
| Franka | 1 (kitchen) | DONE |
| TestArm | 1 (arm) | DONE |
| UR5e | 1 (arm) | DONE |
| RobotiqGripper | 1 (gripper) | DONE |
| **Total** | **17 arm/hand + 8 locomotion = 25** | **COMPLETE** |

## Out of scope (require different approach)

- **OGBench Cube/Scene** (image-based, requires ViT encoder)
- **FetchPickAndPlace** (velocity control, not joint)
- **D4RL/DexMV** (custom control loops)
- **ManiSkill** (Isaac simulator, different physics)

## Next experiments

1. **Train 1M+ data with 10 epochs** (target 90%+ reach task improvement)
2. **Architecture ablation** (deeper SNN vs Transformer)
3. **Apply to OGBench Cube** (with ViT encoder for pixel input)
4. **Loihi deployment** (energy benchmarking)
5. **Paper update** (v3.2 → v4 with 25 env results + 87.5% reach)

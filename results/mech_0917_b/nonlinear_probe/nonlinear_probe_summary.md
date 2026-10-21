# E1: Nonlinear / temporal accessibility probe (cls0917-b)

128 windows/env; contexts: WINDOW (27-frame ctx) and WARM (full-episode forward) on the same t positions. R2 = held-out episode-level 75/25, uniform average over non-constant columns of the 128-d padded target (= native dims here); literal 128-col values in JSON only. Velocity = pseudo-velocity (first difference; npz has no qvel).

MLP: 256-hidden, AdamW, early stopping, 3 seeds (mean +- seed std).

## (i) s_t, pooled over envs: linear vs MLP vs temporal

| model | win linear | win MLP | win MLP std | win temporal | warm linear | warm MLP | warm MLP std | warm temporal |
|---|---|---|---|---|---|---|---|---|
| trace | -1.154 | 0.243 | 0.058 | -0.131 | -1.531 | 0.388 | 0.020 | -0.901 |
| spike | -7.748 | -0.505 | 0.438 | 0.065 | -2.389 | 0.117 | 0.045 | 0.118 |
| membrane | 0.028 | 0.511 | 0.008 | 0.410 | 0.340 | 0.531 | 0.017 | 0.438 |
| hidden_leak | 0.168 | 0.507 | 0.012 | 0.432 | 0.330 | 0.516 | 0.015 | 0.396 |
| lewm | 0.929 | 0.898 | 0.040 | 0.889 | 0.922 | 0.881 | 0.038 | 0.880 |
| gru | 0.647 | 0.557 | 0.009 | -0.269 | 0.637 | 0.557 | 0.011 | 0.172 |
| mlp | 0.712 | 0.611 | 0.010 | 0.715 | 0.694 | 0.612 | 0.012 | 0.695 |
| slif_trace | -0.350 | -0.174 | 0.046 | -5.401 | -0.178 | -0.094 | 0.037 | -2.342 |

## (ii) Headline: linear -> MLP gain on s_t (pooled)

| model | win gain (MLP - linear) | warm gain |
|---|---|---|
| trace | 1.397 | 1.919 |
| spike | 7.243 | 2.506 |
| membrane | 0.483 | 0.191 |
| hidden_leak | 0.340 | 0.186 |
| lewm | -0.031 | -0.041 |
| gru | -0.090 | -0.080 |
| mlp | -0.102 | -0.082 |
| slif_trace | 0.176 | 0.084 |

Controls: LeWM/GRU gains should be small; a large trace gain => information hidden nonlinearly, not discarded.

## (iii) Pseudo-velocity (pooled)

| model | win linear | win MLP | warm linear | warm MLP |
|---|---|---|---|---|
| trace | -1.734 | 0.037 | -2.837 | -0.020 |
| spike | -8.194 | -0.271 | -2.935 | -0.019 |
| membrane | -0.506 | 0.015 | -0.335 | 0.065 |
| hidden_leak | -0.467 | 0.067 | -0.334 | 0.096 |
| lewm | -0.378 | 0.113 | -0.350 | 0.112 |
| gru | 0.190 | 0.127 | 0.203 | 0.131 |
| mlp | -0.138 | 0.120 | -0.069 | 0.107 |
| slif_trace | -0.155 | -0.044 | -0.160 | -0.044 |

## Futures: s_t+5 (pooled)

| model | win linear | win MLP | warm linear | warm MLP |
|---|---|---|---|---|
| trace | -0.765 | 0.251 | -1.152 | 0.385 |
| spike | -6.582 | -0.459 | -2.119 | 0.141 |
| membrane | 0.119 | 0.519 | 0.320 | 0.528 |
| hidden_leak | 0.230 | 0.494 | 0.347 | 0.518 |
| lewm | 0.594 | 0.685 | 0.600 | 0.685 |
| gru | 0.594 | 0.530 | 0.582 | 0.529 |
| mlp | 0.574 | 0.575 | 0.599 | 0.574 |
| slif_trace | -0.330 | -0.156 | -0.166 | -0.073 |

## Futures: s_t+20 (pooled)

| model | win linear | win MLP | warm linear | warm MLP |
|---|---|---|---|---|
| trace | -0.684 | 0.273 | -0.843 | 0.392 |
| spike | -5.367 | -0.376 | -1.933 | 0.143 |
| membrane | 0.132 | 0.518 | 0.262 | 0.537 |
| hidden_leak | 0.264 | 0.514 | 0.347 | 0.534 |
| lewm | 0.360 | 0.589 | 0.377 | 0.589 |
| gru | 0.446 | 0.492 | 0.438 | 0.486 |
| mlp | 0.400 | 0.566 | 0.496 | 0.565 |
| slif_trace | -0.279 | -0.151 | -0.121 | -0.069 |

## Overfitting check (s_t, pooled): train vs test R2

| model | win lin train/test | win MLP train/test | warm lin train/test | warm MLP train/test |
|---|---|---|---|---|
| trace | 0.898 / -1.154 | 0.554 / 0.243 | 0.895 / -1.531 | 0.578 / 0.388 |
| spike | 0.951 / -7.748 | 0.301 / -0.505 | 0.994 / -2.389 | 0.409 / 0.117 |
| membrane | 0.960 / 0.028 | 0.631 / 0.511 | 0.957 / 0.340 | 0.610 / 0.531 |
| hidden_leak | 0.950 / 0.168 | 0.618 / 0.507 | 0.948 / 0.330 | 0.609 / 0.516 |
| lewm | 0.990 / 0.929 | 0.976 / 0.898 | 0.991 / 0.922 | 0.973 / 0.881 |
| gru | 0.806 / 0.647 | 0.639 / 0.557 | 0.812 / 0.637 | 0.641 / 0.557 |
| mlp | 0.894 / 0.712 | 0.675 / 0.611 | 0.900 / 0.694 | 0.673 / 0.612 |
| slif_trace | 0.273 / -0.350 | 0.196 / -0.174 | 0.206 / -0.178 | 0.034 / -0.094 |

## Per-env appendix: s_t R2 (linear / MLP / temporal)

### window

| model | cartpole_2d lin/MLP/temp | cheetah lin/MLP/temp | finger lin/MLP/temp |
|---|---|---|---|
| trace | 0.445 / 0.556 / 0.746 | -2.203 / -0.077 / -0.868 | -1.704 / 0.249 / -0.271 |
| spike | -2.137 / -1.030 / 0.707 | -6.141 / -0.097 / -0.393 | -14.967 / -0.388 / -0.120 |
| membrane | 0.983 / 0.978 / 0.997 | -0.969 / -0.058 / -0.071 | 0.070 / 0.613 / 0.305 |
| hidden_leak | 0.973 / 0.944 / 0.998 | -0.970 / -0.036 / -0.093 | 0.500 / 0.614 / 0.391 |
| lewm | 0.973 / 0.968 / 0.983 | 0.908 / 0.904 / 0.821 | 0.907 / 0.822 / 0.865 |
| gru | 0.967 / 0.892 / 0.974 | 0.349 / 0.100 / 0.432 | 0.624 / 0.679 / -2.212 |
| mlp | 1.000 / 0.997 / 1.000 | 0.422 / 0.130 / 0.550 | 0.716 / 0.704 / 0.595 |
| slif_trace | -0.217 / -0.135 / -1.242 | -0.260 / -0.125 / -6.330 | -0.572 / -0.263 / -8.629 |
### warm

| model | cartpole_2d lin/MLP/temp | cheetah lin/MLP/temp | finger lin/MLP/temp |
|---|---|---|---|
| trace | 0.663 / 0.851 / 0.624 | -3.099 / -0.127 / -1.888 | -2.155 / 0.440 / -1.438 |
| spike | -0.191 / 0.413 / 0.688 | -3.746 / -0.099 / -0.152 | -3.229 / 0.037 / -0.181 |
| membrane | 0.989 / 0.970 / 0.998 | -0.605 / -0.045 / -0.109 | 0.635 / 0.668 / 0.425 |
| hidden_leak | 0.984 / 0.938 / 0.997 | -0.511 / -0.033 / -0.210 | 0.516 / 0.643 / 0.401 |
| lewm | 0.971 / 0.953 / 0.979 | 0.904 / 0.898 / 0.812 | 0.891 / 0.793 / 0.849 |
| gru | 0.971 / 0.872 / 0.982 | 0.352 / 0.102 / 0.432 | 0.587 / 0.697 / -0.898 |
| mlp | 1.000 / 0.997 / 1.000 | 0.368 / 0.135 / 0.479 | 0.714 / 0.704 / 0.606 |
| slif_trace | -0.092 / -0.063 / -1.637 | -0.210 / -0.105 / -2.266 | -0.231 / -0.115 / -3.123 |

## Per-env appendix: velocity R2 (linear / MLP)

### window

| model | cartpole_2d lin/MLP | cheetah lin/MLP | finger lin/MLP |
|---|---|---|---|
| trace | -1.171 / 0.273 | -1.848 / -0.097 | -2.185 / -0.064 |
| spike | -5.776 / -0.443 | -8.176 / -0.093 | -10.630 / -0.277 |
| membrane | 0.173 / 0.157 | -1.474 / -0.078 | -0.216 / -0.034 |
| hidden_leak | 0.340 / 0.263 | -1.324 / -0.055 | -0.419 / -0.008 |
| lewm | 0.355 / 0.439 | -0.664 / -0.100 | -0.824 / 0.002 |
| gru | 0.519 / 0.460 | 0.177 / -0.020 | -0.127 / -0.060 |
| mlp | 0.423 / 0.440 | -0.632 / -0.067 | -0.205 / -0.013 |
| slif_trace | -0.069 / -0.009 | -0.369 / -0.114 | -0.027 / -0.008 |
### warm

| model | cartpole_2d lin/MLP | cheetah lin/MLP | finger lin/MLP |
|---|---|---|---|
| trace | -1.316 / 0.257 | -3.905 / -0.173 | -3.291 / -0.145 |
| spike | -1.885 / 0.065 | -5.146 / -0.085 | -1.775 / -0.038 |
| membrane | 0.335 / 0.303 | -1.188 / -0.096 | -0.151 / -0.013 |
| hidden_leak | 0.297 / 0.369 | -1.165 / -0.062 | -0.134 / -0.019 |
| lewm | 0.382 / 0.437 | -0.653 / -0.101 | -0.780 / 0.001 |
| gru | 0.557 / 0.477 | 0.181 / -0.020 | -0.131 / -0.065 |
| mlp | 0.407 / 0.402 | -0.409 / -0.068 | -0.206 / -0.013 |
| slif_trace | -0.231 / 0.025 | -0.216 / -0.098 | -0.032 / -0.058 |

## Per-env appendix: futures R2, linear / MLP (window ctx)

| model | env | s_t+5 lin/MLP | s_t+20 lin/MLP |
|---|---|---|---|
| trace | cartpole_2d | 0.442 / 0.554 | 0.401 / 0.559 |
| trace | cheetah | -1.691 / -0.071 | -1.296 / -0.052 |
| trace | finger | -1.046 / 0.271 | -1.159 / 0.312 |
| spike | cartpole_2d | -2.255 / -0.955 | -2.732 / -0.800 |
| spike | cheetah | -6.663 / -0.088 | -5.317 / -0.055 |
| spike | finger | -10.829 / -0.333 | -8.053 / -0.273 |
| membrane | cartpole_2d | 0.981 / 0.975 | 0.960 / 0.940 |
| membrane | cheetah | -0.821 / -0.024 | -0.886 / 0.014 |
| membrane | finger | 0.196 / 0.606 | 0.324 / 0.601 |
| hidden_leak | cartpole_2d | 0.973 / 0.940 | 0.964 / 0.924 |
| hidden_leak | cheetah | -0.800 / -0.072 | -0.735 / 0.020 |
| hidden_leak | finger | 0.518 / 0.614 | 0.564 / 0.598 |
| lewm | cartpole_2d | 0.975 / 0.958 | 0.976 / 0.961 |
| lewm | cheetah | -0.126 / 0.241 | -0.723 / 0.018 |
| lewm | finger | 0.931 / 0.857 | 0.828 / 0.788 |
| gru | cartpole_2d | 0.969 / 0.897 | 0.970 / 0.896 |
| gru | cheetah | 0.224 / 0.039 | -0.241 / -0.056 |
| gru | finger | 0.590 / 0.654 | 0.610 / 0.637 |
| mlp | cartpole_2d | 1.000 / 0.997 | 0.996 / 0.991 |
| mlp | cheetah | 0.016 / 0.041 | -0.473 / 0.037 |
| mlp | finger | 0.706 / 0.686 | 0.676 / 0.669 |
| slif_trace | cartpole_2d | -0.225 / -0.138 | -0.246 / -0.145 |
| slif_trace | cheetah | -0.326 / -0.123 | -0.201 / -0.101 |
| slif_trace | finger | -0.440 / -0.206 | -0.389 / -0.206 |

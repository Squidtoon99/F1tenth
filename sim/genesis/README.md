# sim/genesis (historical)

RL training uses NVIDIA Warp in
[`training/f1tenth_sim/`](../../training/f1tenth_sim/). This directory retains
historical notes only; Genesis is not a training dependency.

The observation layout produced by the training environment comes from the shared
contract ([`libs/f1tenth_contract`](../../libs/f1tenth_contract/)), so trained
policies match the on-car inference node.

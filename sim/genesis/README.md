# sim/genesis (historical)

RL training uses the NVIDIA Warp simulator in
[`training/f1tenth_sim/`](../../training/f1tenth_sim/) (ADR 0007). This directory
retains historical notes only; Genesis is not a training dependency.

The observation layout produced by the training environment comes from the shared
contract ([`libs/f1tenth_contract`](../../libs/f1tenth_contract/)), so trained
policies match the on-car inference node.

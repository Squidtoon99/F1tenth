# sim/genesis (historical)

RL training uses the Torch simulator in
[`training/f1tenth_sim/`](../../training/f1tenth_sim/). This directory retains
historical notes only; Genesis is not a training dependency.

This directory holds notes and any small helper assets (e.g. the vehicle URDF used
by the sim) that are worth versioning. Large artifacts stay out of git.

The observation layout produced by the training environment comes from the shared
contract ([`libs/f1tenth_contract`](../../libs/f1tenth_contract/)), so trained
policies match the on-car inference node.

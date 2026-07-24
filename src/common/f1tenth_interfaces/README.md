# f1tenth_interfaces

Message/service definitions shared across the stack — the topic contract that
perception, planning, control, and the racing stacks agree on.

Messages:

- `msg/Observation.msg` — flat observation vector (layout defined in
  [`libs/f1tenth_contract`](../../../libs/f1tenth_contract/)).
- `msg/Action.msg` — normalized policy action.
- `msg/ActuatorCommand.msg` — atomic desired/applied physical actuator command
  for the RL current gate (`/rl/actuator/desired`, `/rl/actuator/applied`).

`ActuatorCommand` carries generation metadata, the originating observation
timestamp, calibrated motor/brake currents and servo position, normalized
longitudinal/steering for policy history, and a source enum (`SOURCE_SAFE`,
`SOURCE_RL`, `SOURCE_TELEOP`, `SOURCE_SAFETY`).

> Standard sensor/command topics (`/scan`, `/odom`, `/drive`, `/sensors/*`) reuse
> upstream message types (`sensor_msgs`, `nav_msgs`, `ackermann_msgs`) and are not
> redefined here.

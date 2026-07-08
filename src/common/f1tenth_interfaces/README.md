# f1tenth_interfaces

Message/service definitions shared across the stack — the topic contract that
perception, planning, control, and the racing stacks agree on.

Current placeholders:

- `msg/Observation.msg` — flat observation vector (layout defined in
  [`libs/f1tenth_contract`](../../../libs/f1tenth_contract/)).
- `msg/Action.msg` — normalized policy action.

> Standard sensor/command topics (`/scan`, `/odom`, `/drive`, `/sensors/*`) reuse
> upstream message types (`sensor_msgs`, `nav_msgs`, `ackermann_msgs`) and are not
> redefined here.

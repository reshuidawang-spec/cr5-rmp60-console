# Robot and Probe Models

- `cr5_with_rmp60.urdf.xacro`: CR5 model with the RMP60 probe assembly.
- `cr5_moveit_with_rmp60.urdf.xacro`: MoveIt-oriented combined model.
- `rmp60_probe_module.xacro`: reusable probe, five styli, and collision geometry.

Model dimensions affect collision checking and measurement reconstruction; update
them only together with geometry validation.

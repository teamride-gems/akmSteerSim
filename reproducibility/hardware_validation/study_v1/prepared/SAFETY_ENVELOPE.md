# Conservative pre-hardware path envelope

This is an engineering screen using the separate slow-actuator kinematic plant after the common hardware safety limiter. It is not physical evidence and must not replace the surveyed geofence or safety supervisor.

- Maximum predicted radius from the run start: `0.750 m`
- Maximum predicted absolute lateral displacement: `0.020 m`
- Maximum predicted absolute heading: `0.043 rad`
- Frozen runtime geofence radius: `1.25 m`
- Required clear course: predicted envelope plus at least 1.0 m physical margin on every side.

The compact profile assumes a centered start in a measured 16 ft × 16 ft unobstructed square and includes the configured zero-command stop packets in the modeled envelope.

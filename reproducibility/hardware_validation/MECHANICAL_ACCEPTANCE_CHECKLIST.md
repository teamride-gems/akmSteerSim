# Frank mechanical and instrumentation acceptance checklist

> This checklist covers qualification only. Passing it does not authorize
> `HW001`, the historical 120-run schedule, or any scientific collection.

Complete and sign this checklist before the ROS 1 live preflight. Record a value or an explicit pass/fail result for every row; blank cells are failures.

## Identification and configuration

- [ ] Robot is physically labeled and recorded as Frank/Franklin; chassis identity is unambiguous.
- [ ] Nvidia AGX Xavier boots Ubuntu 20.04 and ROS Noetic.
- [ ] Current `vesc.yaml`, `joy_teleop.yaml`, and every launch file used for testing have been captured with `scripts/capture_ros1_configuration.py`.
- [ ] The autonomous Ackermann mux input is distinct from or safely arbitrated with teleoperation.
- [ ] Teleoperation override and the independent physical motor e-stop have both been demonstrated on stands.
- [ ] Plaintext credentials previously stored in shared Drive documents have been rotated and removed.

## Required measurements

| Measurement | Recorded value | Method/instrument | Pass criterion |
|---|---:|---|---|
| Wheelbase | ______ m | axle-center to axle-center, 3 repeats | range ≤ 2 mm |
| Mass with test battery | ______ kg | scale, test configuration | documented |
| Left steering limit | ______ rad | wheel angle, not servo command | documented |
| Right steering limit | ______ rad | wheel angle, not servo command | documented |
| Minimum left turn radius | ______ m | low-speed marked circle | documented |
| Minimum right turn radius | ______ m | low-speed marked circle | documented |
| Steering step latency | ______ ms | command/pose timestamps, ≥10 steps | median and p95 documented |
| Speed response latency | ______ ms | command/odometry, ≥10 steps | median and p95 documented |
| Stop distance from 0.5 m/s | ______ m | ≥5 stops on test floor | all within clear margin |
| Stop distance from 1.0 m/s | ______ m | ≥5 stops on test floor | all within clear margin |
| Battery voltage at start/end | ______ / ______ V | logged under load | no cutoff/brownout |

## Drivetrain and steering repair verification

- [ ] Steering servo reaches both directions smoothly without binding, chatter, or missed commands.
- [ ] Front wheels return repeatably to center; center offset has been recorded.
- [ ] Rear motor mount, pinion/spur mesh, gearbox, driveshafts, and wheel fasteners are secure.
- [ ] Slipper clutch/spring is installed, adjusted, and marked so loosening can be detected.
- [ ] Five gradual launches produce no grinding, wheel-free motor spin, clutch slip, or intermittent engagement.
- [ ] Five direction-neutral-stop tests produce no abrupt reversal or drivetrain shock.
- [ ] `vesc.yaml` has exactly one `max_acceleration` value and it is 2.0 m/s².
- [ ] The car remains stopped for 10 s after zero-speed commands and after autonomous-runner exit.
- [ ] No button must be held to run autonomously; the bounded heartbeat test releases the override and automatically restores it when the heartbeat ends.
- [ ] `/vesc/joy` index 6 latches the software e-stop, and joystick disconnection restores the zero-command override no later than 0.30 s after the last message.

## Sensing, localization, and recording

- [ ] `/scan` is live with plausible ranges and no persistent blind sector caused by the chassis.
- [ ] Composed Cartographer `cartographer_map -> base_link` pose is ≥40 Hz, maximum gap ≤0.10 s, and stationary drift ≤0.02 m over the capture interval.
- [ ] IMU, if used, has a secure mount, correct axes, stable timestamps, and no cable strain.
- [ ] Onboard localization system: ______________________________
- [ ] Independent evaluation system: ____________________________
- [ ] ArUco/external pose accuracy and update rate were measured, or its unavailability was documented before outcomes.
- [ ] If independent tracking is used, onboard and independent poses agree in translation, heading, direction, and time alignment during a slow figure-eight; otherwise mark this N/A under the prospectively documented Cartographer-only limitation.
- [ ] rosbag1 contains command, odometry, authorization, runner-heartbeat, e-stop, scan, and configured VESC feedback topics.
- [ ] Independent video covers the entire course and visibly identifies run ID.

## Course and safety release

- [ ] Start pose is taped and captured after the robot has remained stationary.
- [ ] Obstacle-free radius is at least 6.0 m and was measured rather than estimated.
- [ ] No spectators can enter the test area during motion.
- [ ] Operator: __________________  Date/time: __________________
- [ ] Safety supervisor: __________  Date/time: __________________
- [ ] Mechanical lead: ____________  Date/time: __________________

**Release decision:** [ ] PASS FOR STANDS ONLY  [ ] PASS FOR GROUND PILOT  [ ] FAIL

Any failure blocks further qualification at the affected level. Passing this
checklist authorizes only the marked engineering level; it does not authorize a
scientific study. Qualification data remain excluded from paper analysis unless
a later prospective protocol explicitly defines their use.

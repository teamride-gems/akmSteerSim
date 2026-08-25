# Frank stationary ROS capture: 2026-08-25

This is a pre-motion interface capture, not a scientific outcome. It was used
only to resolve Frank's ROS 1 transport and logging configuration before the
engineering pilots.

## Source artifact

- File: `frank_cartographer_2026-08-25_1517.bag`
- SHA-256: `3849286E683F523FC179AF34D79D9F6130D9FE07D12A69943009F702EE49D035`
- Size: 12,225,448 bytes
- Recorded interval: 2026-08-25 19:17:29.634–19:17:39.212 UTC
- Duration: 9.577916 seconds
- Messages: 9,721 across 44 ROS bag connections

The raw bag is retained outside the Git repository because it is a binary robot
capture. The hash above identifies the reviewed artifact.

## Confirmed interfaces

| Role | Topic | Type | Observed rate |
|---|---|---|---:|
| Cartographer transforms | `/tf` | `tf2_msgs/TFMessage` | 198.45 Hz |
| Static transforms | `/tf_static` | `tf2_msgs/TFMessage` | latched |
| Wheel odometry | `/vesc/odom` | `nav_msgs/Odometry` | 50.01 Hz |
| Joystick | `/vesc/joy` | `sensor_msgs/Joy` | 20.07 Hz |
| LiDAR | `/scan` | `sensor_msgs/LaserScan` | 39.98 Hz |
| IMU | `/imu` | `sensor_msgs/Imu` | 20.00 Hz |
| VESC state | `/vesc/sensors/core` | `vesc_msgs/VescStateStamped` | 50.01 Hz |
| Low-level mux output | `/vesc/low_level/ackermann_cmd_mux/output` | `ackermann_msgs/AckermannDriveStamped` | 50.00 Hz |

Cartographer published both `cartographer_map -> cartographer_odom` and
`cartographer_odom -> base_link` in each dynamic TF message. The experiment
adapter therefore composes those transforms for fixed-world position and
heading while using `/vesc/odom` for speed and yaw rate.

The stationary capture selected `Default` at the high-level mux and
`Teleoperation` at the low-level mux. All Ackermann speed and steering commands
were zero. The VESC reported zero speed, zero duty cycle, and fault code zero.
Input voltage ranged from 12.2 to 12.4 V. The centered servo command was 0.5304.

Cartographer's composed `cartographer_map -> base_link` pose varied by about
5.6 mm in x, 4.2 mm in y, and 0.00066 rad in yaw over the capture. This is an
interface observation only; it is not a localization-accuracy claim because no
independent ground truth was recorded.

## Unresolved gates

The bag cannot establish the active `vesc.yaml` `max_acceleration` parameter,
and no joystick button was pressed, so it cannot identify the physical control
mapped to button index 6. The configured wheelbase value of approximately
0.250 m also remains subject to the protocol's physical-measurement gate.

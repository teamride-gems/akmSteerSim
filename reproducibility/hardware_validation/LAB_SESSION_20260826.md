# Frank lab session record: 2026-08-26

## Status

Engineering evidence only. These bags are not binding scientific outcomes and
do not authorize `HW001`.

Analyzed inputs:

- `frank_ground_preflight_aborted_20260826T220638Z.bag`
- `frank_ground_crawl_20260826T220844Z.bag`

The aborted-preflight bag is stored in the RiDE Drive handoff under
`01 Lab Data`, with SHA-256
`E793C92B1EBC754C61E2C26DC6FBFE8E48ACA426A5ADC6EB97A32C997ACF3407`.
The ground-crawl bag was analyzed during the original review but is no longer
available on the local machine and was not uploaded. Recover the original from
the lab team before treating the handoff as a complete raw-data archive.

## Aborted preflight bag

- Duration: 7.816 s.
- The software e-stop remained asserted.
- The low-level mux remained on `Safety`.
- All recorded speed and steering commands were zero.
- Wheel odometry and VESC speed remained zero.
- LiDAR, IMU, Cartographer TF, joystick, VESC state, servo state, diagnostics,
  and safety topics were present.
- The bag does not contain the application-level failure reason. The terminal
  output or preflight artifact is still required.

This is safe fail-closed behavior, not a passed preflight.

## Ground-crawl bag

- Duration: 125.696 s.
- LiDAR recorded at approximately 40 Hz with 1,081 ranges per scan and about
  96% in-range returns.
- Wheel odometry and VESC state recorded at approximately 50 Hz.
- IMU, joystick, deadman, and e-stop state recorded at approximately 20 Hz.
- Cartographer published the expected
  `cartographer_map -> cartographer_odom -> base_link` transforms.
- The e-stop reset from asserted to clear at 17.525 s.
- Runner heartbeat was true from 37.132 s to 41.981 s.
- The low-level mux transitioned from `Safety` to `Navigation`, then returned
  to `Safety` about 44 ms after the heartbeat became false.
- The derived authorization became true after the bridge's clearance period
  and returned false when the heartbeat stopped.
- All recorded speed commands, steering commands, wheel speeds, and VESC motor
  speeds were zero. The servo command remained at the neutral value 0.5304.
- No joystick button press occurred in the recording.
- Cartographer estimated about 4.09 m of accumulated path and 0.46 m net
  displacement while wheel odometry stayed fixed. This is compatible with the
  car being carried or pushed, or with localization motion independent of
  wheel feedback; it is not evidence of powered driving.
- VESC fault code remained zero. Input voltage ranged from 11.8 V to 12.1 V.

## Supported conclusions

- Core sensor topics and recording are functional.
- Cartographer's required TF chain is available.
- VESC feedback is available and reported no fault in these bags.
- The autonomous heartbeat can release and restore mux authorization.
- Heartbeat loss returned the mux to `Safety` promptly.

## Not yet verified

- Nonzero autonomous speed and steering commands reach Frank.
- Steering sign, scale, rate, and physical response.
- Propulsion response and wheel-odometry behavior under powered motion.
- Joystick button index 6 latches the software e-stop.
- Joystick disconnection restores `Safety` within the required timeout.
- The full live preflight passes.
- The 0.20 m/s stands and 0.50 m/s ground engineering pilots pass.
- A normal teleoperation bag contains commands, motion, LiDAR, odometry, TF,
  and actuator feedback together.

## Next permitted lab work

1. Capture the aborted-preflight terminal output and generated attempt folder.
2. Run the bounded stands heartbeat check with a deliberate button-6 stop.
3. Run it again with deliberate joystick disconnection.
4. After both pass, run the 0.20 m/s stands engineering pilot and verify
   nonzero commands and response.
5. Only after that passes, run the 0.50 m/s ground engineering pilot.
6. With the study runner and safety bridge exited, record a short normal
   teleoperation calibration bag.

Do not run the blinded 120-run schedule. A new closed-loop scientific protocol
must be designed separately.

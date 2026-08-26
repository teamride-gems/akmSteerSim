import math
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import yaml

from hardware_study.ros1_adapter import Ros1AckermannAdapter
from hardware_study.ros1_runtime import (
    ros1_bag_command,
    ros1_bag_topics,
    validate_ros1_site,
    verify_operator_prepared,
    verify_operator_paths,
    verify_ros1_amendment,
    AMENDMENT_ID,
)
from hardware_study.integrity import sha256_file
from scripts.capture_ros1_configuration import find_named_values
from scripts.capture_hardware_site_ros1 import restore_template_course_fields
from scripts.analyze_hardware_study_ros1 import classify_ros1_evidence
from scripts.prepare_ros1_operator_package import build_operator_package
from scripts.ros1_hardware_safety_bridge import SafetyState
from scripts.run_hardware_study_ros1 import translate_arguments


ROOT = Path(__file__).resolve().parents[1]


class FakePublisher:
    def __init__(self):
        self.messages = []
        self.closed = False

    def publish(self, message):
        self.messages.append(message)

    def unregister(self):
        self.closed = True


class FakeSubscriber:
    def __init__(self, topic, callback):
        self.topic = topic
        self.callback = callback
        self.closed = False

    def unregister(self):
        self.closed = True


class FakeDriveMessage:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None, frame_id="")
        self.drive = SimpleNamespace(steering_angle=0.0, speed=0.0)


class FakeRospy:
    class Core:
        @staticmethod
        def is_initialized():
            return True

    class Time:
        @staticmethod
        def now():
            return "fake_ros_time"

    core = Core()

    def __init__(self):
        self.publisher = FakePublisher()
        self.subscribers = []

    def init_node(self, *args, **kwargs):
        raise AssertionError("already initialized")

    def Publisher(self, *args, **kwargs):
        return self.publisher

    def Subscriber(self, topic, message_type, callback, queue_size):
        del message_type, queue_size
        subscriber = FakeSubscriber(topic, callback)
        self.subscribers.append(subscriber)
        return subscriber


def fake_runtime():
    return SimpleNamespace(
        rospy=FakeRospy(),
        AckermannDriveStamped=FakeDriveMessage,
        Odometry=object,
        Bool=object,
        JointState=object,
        BatteryState=object,
        TFMessage=object,
    )


def valid_site():
    return {
        "platform": {"ros_version": 1, "ros_distro": "noetic"},
        "topics": {
            "drive": "/drive",
            "odometry": "/odom",
            "deadman": "/deadman",
            "estop": "/estop",
            "safety_override": "/safety_override",
            "joint_states": None,
            "battery_state": None,
            "additional_bag_topics": ["/scan", "/scan"],
        },
        "frames": {"command_frame_id": "base_link"},
        "steering_feedback": {
            "wheelbase_m": 0.33,
            "minimum_speed_for_kinematic_estimate_mps": 0.2,
            "steering_joint_name": None,
        },
        "calibration": {"wheelbase_measured": True},
        "vehicle_limits": {"controller_max_acceleration_mps2": 2.0},
        "course": {
            "localization_system": "cartographer",
            "evaluation_ground_truth_system": "none_onboard_cartographer_only",
        },
        "safety_bridge": {
            "joy_topic": "/joy",
            "deadman_button_index": 5,
            "estop_button_index": 6,
            "deadman_clearance_seconds": 1.0,
        },
        "rosbag": {"compression": "none"},
    }


class Ros1AmendmentTests(unittest.TestCase):
    def test_ros1_adapter_preserves_command_and_telemetry_contract(self):
        runtime = fake_runtime()
        adapter = Ros1AckermannAdapter(valid_site(), runtime=runtime)
        adapter._deadman_callback(SimpleNamespace(data=True))
        adapter._estop_callback(SimpleNamespace(data=False))
        odometry = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: 12.5)),
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=1.0, y=2.0),
                    orientation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
                )
            ),
            twist=SimpleNamespace(
                twist=SimpleNamespace(
                    linear=SimpleNamespace(x=1.0, y=0.0),
                    angular=SimpleNamespace(z=0.5),
                )
            ),
        )
        adapter._odom_callback(odometry)
        telemetry = adapter.latest_telemetry()
        self.assertEqual(telemetry["source_stamp_s"], 12.5)
        self.assertEqual(telemetry["deadman"], True)
        self.assertEqual(telemetry["estop"], False)
        self.assertAlmostEqual(telemetry["steering_rad"], math.atan(0.33 * 0.5))
        adapter.publish(0.12, 0.8, 0.05)
        sent = runtime.rospy.publisher.messages[-1]
        self.assertEqual(sent.header.stamp, "fake_ros_time")
        self.assertEqual(sent.header.frame_id, "base_link")
        self.assertAlmostEqual(sent.drive.steering_angle, 0.12)
        self.assertAlmostEqual(sent.drive.speed, 0.8)
        adapter.close()
        self.assertTrue(runtime.rospy.publisher.closed)

    def test_ros1_adapter_composes_cartographer_tf_with_wheel_odometry(self):
        runtime = fake_runtime()
        site = valid_site()
        site["topics"]["localization_tf"] = "/tf"
        site["frames"] = {
            "command_frame_id": "base_link",
            "odometry_frame_id": "cartographer_map",
            "localization_odom_frame_id": "cartographer_odom",
            "base_frame_id": "base_link",
            "localization_stale_seconds": 0.1,
        }
        adapter = Ros1AckermannAdapter(site, runtime=runtime)
        adapter._deadman_callback(SimpleNamespace(data=True))
        adapter._estop_callback(SimpleNamespace(data=False))
        odometry = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: 12.0)),
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=99.0, y=99.0),
                    orientation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
                )
            ),
            twist=SimpleNamespace(
                twist=SimpleNamespace(
                    linear=SimpleNamespace(x=1.0, y=0.0),
                    angular=SimpleNamespace(z=0.5),
                )
            ),
        )
        adapter._odom_callback(odometry)

        def transform(parent, child, x, y, yaw, stamp):
            return SimpleNamespace(
                header=SimpleNamespace(
                    frame_id=parent,
                    stamp=SimpleNamespace(to_sec=lambda: stamp),
                ),
                child_frame_id=child,
                transform=SimpleNamespace(
                    translation=SimpleNamespace(x=x, y=y),
                    rotation=SimpleNamespace(
                        w=math.cos(yaw / 2.0),
                        x=0.0,
                        y=0.0,
                        z=math.sin(yaw / 2.0),
                    ),
                ),
            )

        adapter._tf_callback(
            SimpleNamespace(
                transforms=[
                    transform("cartographer_map", "cartographer_odom", 1.0, 2.0, math.pi / 2.0, 12.5),
                    transform("cartographer_odom", "base_link", 3.0, 4.0, 0.1, 12.5),
                ]
            )
        )
        telemetry = adapter.latest_telemetry()
        self.assertAlmostEqual(telemetry["x_m"], -3.0)
        self.assertAlmostEqual(telemetry["y_m"], 5.0)
        self.assertAlmostEqual(telemetry["yaw_rad"], math.pi / 2.0 + 0.1)
        self.assertEqual(telemetry["pose_source"], "composed_tf")
        self.assertAlmostEqual(telemetry["speed_mps"], 1.0)

    def test_safety_bridge_starts_latched_and_fails_closed(self):
        state = SafetyState(deadman_index=0, estop_index=1, stale_seconds=0.25)
        self.assertEqual(state.snapshot(now=0.0), (False, True))
        state.update([0, 0], now=1.0)
        self.assertEqual(state.reset(now=1.0)[0], True)
        state.update([1, 0], now=1.1)
        self.assertEqual(state.snapshot(now=1.1), (True, False))
        self.assertEqual(state.snapshot(now=1.4), (False, False))
        state.update([0, 1], now=2.0)
        self.assertEqual(state.snapshot(now=2.0), (False, True))

    def test_safety_bridge_clears_mux_before_asserting_deadman(self):
        state = SafetyState(
            deadman_index=0,
            estop_index=1,
            stale_seconds=2.0,
            deadman_clearance_seconds=1.0,
        )
        state.update([0, 0], now=1.0)
        self.assertTrue(state.stop_override_required(now=1.0))
        self.assertTrue(state.reset(now=1.0)[0])
        state.update([1, 0], now=1.1)
        self.assertFalse(state.stop_override_required(now=1.1))
        self.assertEqual(state.snapshot(now=2.0), (False, False))
        self.assertEqual(state.snapshot(now=2.1), (True, False))
        state.update([0, 0], now=2.2)
        self.assertTrue(state.stop_override_required(now=2.2))
        self.assertEqual(state.snapshot(now=2.2), (False, False))

    def test_safety_bridge_requires_deadman_button_alone(self):
        state = SafetyState(
            deadman_index=1,
            estop_index=2,
            stale_seconds=1.0,
            deadman_clearance_seconds=0.0,
        )
        state.update([0, 0, 0], now=1.0)
        self.assertTrue(state.reset(now=1.0)[0])
        state.update([1, 1, 0], now=1.1)
        self.assertEqual(state.snapshot(now=1.1), (False, False))
        self.assertTrue(state.stop_override_required(now=1.1))
        state.update([0, 1, 0], now=1.2)
        self.assertEqual(state.snapshot(now=1.2), (True, False))
        self.assertFalse(state.stop_override_required(now=1.2))

    def test_site_gate_rejects_unmeasured_or_excess_acceleration(self):
        site = valid_site()
        validate_ros1_site(site)
        site["calibration"]["wheelbase_measured"] = False
        with self.assertRaises(RuntimeError):
            validate_ros1_site(site)
        site = valid_site()
        site["vehicle_limits"]["controller_max_acceleration_mps2"] = 2.1
        with self.assertRaises(RuntimeError):
            validate_ros1_site(site)

    def test_site_gate_requires_confirmed_controls_and_ground_truth_declaration(self):
        site = valid_site()
        site["safety_bridge"]["deadman_button_index"] = 4
        with self.assertRaises(RuntimeError):
            validate_ros1_site(site)
        site = valid_site()
        site["course"].pop("evaluation_ground_truth_system")
        with self.assertRaises(RuntimeError):
            validate_ros1_site(site)

    def test_rosbag1_command_is_bounded_and_deduplicated(self):
        site = valid_site()
        topics = ros1_bag_topics(site)
        self.assertEqual(topics.count("/scan"), 1)
        with tempfile.TemporaryDirectory() as temporary:
            command = ros1_bag_command(Path(temporary), site)
        self.assertEqual(command[:3], ["rosbag", "record", "-O"])
        self.assertIn("/drive", command)
        self.assertIn("/odom", command)

    def test_vesc_acceleration_parser_requires_explicit_value(self):
        self.assertEqual(
            find_named_values({"vesc": {"max_acceleration": 1.5}}, "max_acceleration"),
            [1.5],
        )

    def test_ros1_wrapper_translates_only_explicit_ros1(self):
        translated = translate_arguments(["--run-id", "HW001", "--adapter", "ros1"])
        self.assertIn("ros2", translated)
        self.assertIn("local_hardware_site_ros1.yaml", translated)
        self.assertIn(
            "reproducibility/hardware_validation/study_v1/operator_prepared",
            translated,
        )
        with self.assertRaises(ValueError):
            translate_arguments(["--run-id", "HW001", "--adapter", "ros2"])

    def test_ros1_template_is_fail_closed(self):
        template = yaml.safe_load(
            (ROOT / "configs/hardware_site_ros1_template.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(template["platform"]["ros_distro"], "noetic")
        self.assertFalse(template["calibration"]["wheelbase_measured"])
        self.assertEqual(
            template["topics"]["drive"],
            "/vesc/high_level/ackermann_cmd_mux/input/nav_0",
        )
        self.assertEqual(template["topics"]["localization_tf"], "/tf")
        self.assertEqual(
            template["vehicle_limits"]["controller_max_acceleration_mps2"],
            2.0,
        )
        self.assertEqual(template["safety_bridge"]["estop_button_index"], 6)
        self.assertEqual(template["safety_bridge"]["deadman_button_index"], 5)

    def test_ros1_site_capture_preserves_ground_truth_declaration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "template.yaml"
            output = root / "output.yaml"
            template.write_text(
                yaml.safe_dump(
                    {
                        "course": {
                            "surveyed_clear_radius_m": "REPLACE_WITH_RADIUS",
                            "localization_system": "cartographer_tf",
                            "evaluation_ground_truth_system": "none_onboard_cartographer_only",
                        }
                    }
                ),
                encoding="utf-8",
            )
            output.write_text(
                yaml.safe_dump(
                    {
                        "course": {
                            "surveyed_clear_radius_m": 6.0,
                            "localization_system": "cartographer_tf",
                        }
                    }
                ),
                encoding="utf-8",
            )
            restore_template_course_fields(template, output)
            captured = yaml.safe_load(output.read_text(encoding="utf-8"))
            self.assertEqual(captured["course"]["surveyed_clear_radius_m"], 6.0)
            self.assertEqual(
                captured["course"]["evaluation_ground_truth_system"],
                "none_onboard_cartographer_only",
            )

    def test_amendment_and_base_freeze_hashes_verify(self):
        amendment = verify_ros1_amendment(ROOT)
        self.assertEqual(amendment["amendment_id"], AMENDMENT_ID)
        self.assertFalse(amendment["physical_outcomes_observed_before_amendment"])

    def test_operator_package_requires_sealed_key_to_be_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            visible = prepared / "visible.json"
            visible.write_text("{}", encoding="utf-8")
            key = prepared / "condition_key.json"
            key.write_text("{}", encoding="utf-8")
            manifest = {
                "files": [
                    {"path": "visible.json", "sha256": sha256_file(visible)},
                    {"path": "condition_key.json", "sha256": sha256_file(key)},
                ]
            }
            (prepared / "PREPARED_MANIFEST.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            key.unlink()
            self.assertEqual(verify_operator_prepared(prepared), manifest)
            self.assertTrue(
                all(row["passed"] for row in verify_operator_paths(manifest["files"], prepared))
            )
            key.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_operator_prepared(prepared)

    def test_generated_operator_package_contains_only_opaque_conditions(self):
        semantic_conditions = {
            "clean_a",
            "clean_b",
            "direct",
            "innovation_gate",
            "timing_placebo",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "operator_prepared"
            build_operator_package(
                ROOT / "reproducibility/hardware_validation/study_v1/prepared",
                output,
            )
            verify_operator_prepared(output)
            schedule = json.loads(
                (output / "machine_schedule.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                all(row["condition"] == row["condition_code"] for row in schedule)
            )
            envelope_conditions = {
                row.split(",")[4]
                for row in (output / "safety_envelope.csv")
                .read_text(encoding="utf-8")
                .splitlines()[1:]
            }
            self.assertEqual(envelope_conditions, {"A", "B", "C", "D", "E"})
            for path in output.rglob("*"):
                if path.is_file():
                    text = path.read_text(encoding="utf-8")
                    self.assertFalse(
                        any(
                            f'"{condition}"' in text
                            or f',{condition},' in text
                            for condition in semantic_conditions
                        ),
                        str(path),
                    )

    def test_ros1_analysis_labels_only_ros1_outcomes_as_physical(self):
        result = {
            "adapter_names": ["ros1_ackermann_noetic"],
            "evidence_class": "ENGINEERING_MOCK_ONLY",
            "invalid_reasons": [],
            "verdict": "NOT_REPRODUCED",
        }
        classified = classify_ros1_evidence(result)
        self.assertEqual(classified["evidence_class"], "PHYSICAL_HARDWARE")

        mixed = {
            "adapter_names": ["ros1_ackermann_noetic", "mock"],
            "evidence_class": "ENGINEERING_MOCK_ONLY",
            "invalid_reasons": [],
            "verdict": "NOT_REPRODUCED",
        }
        classified = classify_ros1_evidence(mixed)
        self.assertEqual(classified["verdict"], "INVALID")
        self.assertIn("mixed_adapter_evidence", classified["invalid_reasons"])


if __name__ == "__main__":
    unittest.main()

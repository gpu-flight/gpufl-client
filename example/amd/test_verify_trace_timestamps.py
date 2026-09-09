"""Portable regression tests for the example capture validator."""
import unittest
from verify_trace_timestamps import validate


class CaptureValidationTest(unittest.TestCase):
    def events(self):
        epoch = 1788928712000000000
        return [
            {"type": "job_start", "session_id": "test", "ts_ns": epoch},
            {"type": "shutdown", "ts_ns": epoch + 10000000},
            {"type": "kernel_event_batch", "base_time_ns": epoch,
             "columns": ["dt_ns", "duration_ns"], "rows": [[1000, 100]]},
        ]

    def test_rejects_profiler_relative_timestamps(self):
        for kind in ("kernel_event_batch", "memcpy_event_batch",
                     "memory_alloc_event_batch", "synchronization_event_batch"):
            with self.subTest(kind=kind):
                events = self.events()
                events[-1]["type"] = kind
                validate(events)
                events[-1]["base_time_ns"] = 491751431486779
                with self.assertRaisesRegex(ValueError, "outside the session clock"):
                    validate(events)

    def test_static_batch_must_not_be_broadcast_to_every_channel(self):
        batch = {"type": "profile_sample_batch", "session_id": "test", "batch_id": 1000001,
                 "columns": ["sample_kind"], "rows": [[2]]}
        validate(self.events() + [batch])
        with self.assertRaisesRegex(ValueError, "Duplicate static ISA"):
            validate(self.events() + [batch, batch, batch, batch])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
import pytest

import scripts.promote_vision_evidence as promotion_gate
from scripts.promote_vision_evidence import (
    _dotted_differences,
    _replace_pair_with_rollback,
    main,
    promote_vision_evidence,
    semantic_differences,
    sha256_tree,
    snapshot_protected_data,
    traffic_state_semantic_view,
    verify_protected_data,
)
from vision.frame_evidence import (
    RepresentativeFrame,
    TrackEvidence,
    write_frame_evidence,
)


DIRECTIONS = ("north", "south", "east", "west")


def _state() -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "scenario_id": "demo_001",
        "duration_sec": 1,
        "approaches": {
            direction: {
                "observed": direction in ("north", "south"),
                "flow_vph": 400,
                "queue_est": None,
                "vehicle_mix": {
                    "car": 1,
                    "bus": 0,
                    "truck": 0,
                    "motorcycle": 0,
                },
            }
            for direction in DIRECTIONS
        },
        "turning_ratio": {
            direction: {"left": 0.15, "straight": 0.70, "right": 0.15}
            for direction in DIRECTIONS
        },
        "flow_profile": {
            "north": [400],
            "south": [400],
            "east": [],
            "west": [],
        },
        "profile_bins_sec": 5,
        "source": {
            "video": "data/videos/demo.mp4",
            "fps": 10,
            "frames": 10,
            "duration_sec": 1,
            "analyzed_at": "2026-08-19T10:00:00",
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _set_nested(value: Any, path: tuple[object, ...], replacement: Any) -> None:
    target = value
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = replacement


@pytest.fixture
def evidence_root(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "repo"
    reference_state = _state()

    formal_state = root / "data/traffic_states/demo_001.json"
    _write_json(formal_state, reference_state)
    arena = root / "data/results/arena_summary.csv"
    arena.parent.mkdir(parents=True, exist_ok=True)
    arena.write_bytes(b"strategy,score\nfixed,1\n")
    experiment_result = root / "data/results/experiments/run/result.json"
    _write_json(experiment_result, {"score": 1})

    video = root / "data/videos/demo.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"formal-video")
    roi_config = {"count_lines": {"north": [[0, 10], [29, 10]]}}
    roi = root / "data/videos/demo_roi.json"
    _write_json(roi, roi_config)

    destination_frame = root / "figures/fig_vision_annotated_frame.png"
    destination_meta = root / "figures/fig_vision_annotated_frame.meta.json"
    destination_frame.parent.mkdir(parents=True, exist_ok=True)
    destination_frame.write_bytes(b"old-frame")
    destination_meta.write_bytes(b"old-meta")

    candidate_dir = root / "candidate"
    candidate_state = candidate_dir / "traffic_state.json"
    candidate_value = copy.deepcopy(reference_state)
    candidate_value["source"]["analyzed_at"] = "2026-08-19T10:01:00"
    _write_json(candidate_state, candidate_value)

    representative = RepresentativeFrame(
        annotated_bgr=np.full((20, 30, 3), 127, dtype=np.uint8),
        frame_index=2,
        timestamp_sec=0.2,
        fps=10,
        tracks=(
            TrackEvidence(
                track_id=5,
                class_name="car",
                bbox_xyxy=(5.0, 6.0, 15.0, 16.0),
                trace_points=((8.0, 8.0),),
            ),
        ),
    )
    candidate_frame = candidate_dir / "frame.png"
    candidate_meta = candidate_dir / "frame.meta.json"
    write_frame_evidence(
        representative,
        frame_path=candidate_frame,
        meta_path=candidate_meta,
        video_path=video,
        roi_config=roi_config,
    )
    metadata = json.loads(candidate_meta.read_text(encoding="utf-8"))
    metadata["video"]["path"] = "data/videos/demo.mp4"
    _write_json(candidate_meta, metadata)

    return {
        "root": root,
        "formal_state": formal_state,
        "arena": arena,
        "experiment_result": experiment_result,
        "destination_frame": destination_frame,
        "destination_meta": destination_meta,
        "candidate_state": candidate_state,
        "candidate_frame": candidate_frame,
        "candidate_meta": candidate_meta,
        "snapshot": snapshot_protected_data(root.resolve()),
    }


def _destination_bytes(paths: dict[str, Any]) -> tuple[bytes, bytes]:
    return (
        paths["destination_frame"].read_bytes(),
        paths["destination_meta"].read_bytes(),
    )


def _raw_replacement_pair(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "source_frame": tmp_path / "source-frame",
        "source_meta": tmp_path / "source-meta",
        "destination_frame": tmp_path / "destination-frame",
        "destination_meta": tmp_path / "destination-meta",
    }
    paths["source_frame"].write_bytes(b"new-frame")
    paths["source_meta"].write_bytes(b"new-meta")
    paths["destination_frame"].write_bytes(b"old-frame")
    paths["destination_meta"].write_bytes(b"old-meta")
    return paths


def _raw_destination_bytes(paths: dict[str, Path]) -> tuple[bytes, bytes]:
    return (
        paths["destination_frame"].read_bytes(),
        paths["destination_meta"].read_bytes(),
    )


def _transaction_artifacts(tmp_path: Path) -> list[Path]:
    return sorted(
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(".")
        and path.suffix in {".tmp", ".backup", ".restore"}
    )


def test_traffic_state_semantic_view_is_the_exact_scientific_projection() -> None:
    state = _state()

    assert traffic_state_semantic_view(state) == {
        "schema_version": "1.1",
        "duration_sec": 1,
        "approaches": {
            direction: {
                "observed": direction in ("north", "south"),
                "flow_vph": 400,
                "queue_est": None,
                "vehicle_mix": {
                    "car": 1,
                    "bus": 0,
                    "truck": 0,
                    "motorcycle": 0,
                },
            }
            for direction in DIRECTIONS
        },
        "turning_ratio": state["turning_ratio"],
        "flow_profile": state["flow_profile"],
        "profile_bins_sec": 5,
        "source": {"fps": 10, "frames": 10, "duration_sec": 1},
    }


@pytest.mark.parametrize(
    ("function", "path_parameters"),
    [
        (sha256_tree, ("path",)),
        (snapshot_protected_data, ("root",)),
        (verify_protected_data, ("root",)),
        (
            promote_vision_evidence,
            (
                "root",
                "candidate_state_path",
                "candidate_frame_path",
                "candidate_meta_path",
            ),
        ),
    ],
)
def test_public_path_parameters_have_exact_path_annotations(
    function: Any,
    path_parameters: tuple[str, ...],
) -> None:
    hints = get_type_hints(function)

    assert {parameter: hints[parameter] for parameter in path_parameters} == {
        parameter: Path for parameter in path_parameters
    }


@pytest.mark.parametrize(
    "missing_path",
    [
        ("source", "frames"),
        ("approaches", "north", "flow_vph"),
    ],
)
def test_semantic_comparison_rejects_mutually_missing_required_field(
    missing_path: tuple[object, ...],
) -> None:
    reference = _state()
    candidate = copy.deepcopy(reference)
    for value in (reference, candidate):
        target = value
        for component in missing_path[:-1]:
            target = target[component]
        target.pop(missing_path[-1])

    with pytest.raises(KeyError, match=str(missing_path[-1])):
        semantic_differences(reference, candidate)


def test_semantic_differences_ignore_run_identity_and_provenance_paths() -> None:
    reference = _state()
    candidate = copy.deepcopy(reference)
    candidate["source"]["analyzed_at"] = "later"
    candidate["source"]["video"] = "outputs/candidate/demo.mp4"

    assert semantic_differences(reference, candidate) == []


@pytest.mark.parametrize(
    ("path", "replacement", "expected_path"),
    [
        (("duration_sec",), 2, "duration_sec"),
        (("approaches", "north", "flow_vph"), 999, "approaches.north.flow_vph"),
        (("approaches", "east", "observed"), True, "approaches.east.observed"),
        (("turning_ratio", "north", "left"), 0.2, "turning_ratio.north.left"),
        (("flow_profile", "north", 0), 401, "flow_profile.north[0]"),
        (("profile_bins_sec",), 10, "profile_bins_sec"),
        (("source", "fps"), 11, "source.fps"),
        (("source", "frames"), 11, "source.frames"),
        (("source", "duration_sec"), 2, "source.duration_sec"),
    ],
)
def test_semantic_differences_report_exact_scientific_path(
    path: tuple[object, ...],
    replacement: Any,
    expected_path: str,
) -> None:
    reference = _state()
    candidate = copy.deepcopy(reference)
    _set_nested(candidate, path, replacement)

    assert expected_path in semantic_differences(reference, candidate)


def test_dotted_comparator_reports_missing_keys_list_lengths_and_elements() -> None:
    assert _dotted_differences(
        {"mapping": {"kept": 1, "removed": 2}},
        {"mapping": {"kept": 3, "added": 4}},
    ) == ["mapping.added", "mapping.kept", "mapping.removed"]
    assert _dotted_differences({"values": [1]}, {"values": [1, 2]}) == [
        "values"
    ]
    assert _dotted_differences({"values": [1]}, {"values": [2]}) == [
        "values[0]"
    ]


def test_protected_snapshot_detects_experiment_change(
    evidence_root: dict[str, Any],
) -> None:
    verify_protected_data(evidence_root["root"], evidence_root["snapshot"])
    _write_json(evidence_root["experiment_result"], {"score": 2})

    with pytest.raises(ValueError, match=r"protected data changed: .*data/results/experiments"):
        verify_protected_data(evidence_root["root"], evidence_root["snapshot"])


def test_protected_tree_digest_detects_file_rename(
    evidence_root: dict[str, Any],
) -> None:
    renamed = evidence_root["experiment_result"].with_name("renamed.json")
    evidence_root["experiment_result"].rename(renamed)

    with pytest.raises(ValueError, match=r"data/results/experiments"):
        verify_protected_data(evidence_root["root"], evidence_root["snapshot"])


def test_promote_rejects_semantic_change_without_touching_formal_pair(
    evidence_root: dict[str, Any],
) -> None:
    candidate = json.loads(
        evidence_root["candidate_state"].read_text(encoding="utf-8")
    )
    candidate["approaches"]["north"]["flow_vph"] = 999
    _write_json(evidence_root["candidate_state"], candidate)
    before = _destination_bytes(evidence_root)

    with pytest.raises(
        ValueError,
        match=r"^TrafficState semantic mismatch: approaches\.north\.flow_vph$",
    ):
        promote_vision_evidence(
            root=evidence_root["root"],
            candidate_state_path=evidence_root["candidate_state"],
            candidate_frame_path=evidence_root["candidate_frame"],
            candidate_meta_path=evidence_root["candidate_meta"],
            protected_snapshot=evidence_root["snapshot"],
        )

    assert _destination_bytes(evidence_root) == before


def test_promote_rejects_changed_protected_file_without_touching_formal_pair(
    evidence_root: dict[str, Any],
) -> None:
    evidence_root["arena"].write_bytes(b"changed")
    before = _destination_bytes(evidence_root)

    with pytest.raises(ValueError, match=r"data/results/arena_summary\.csv"):
        promote_vision_evidence(
            root=evidence_root["root"],
            candidate_state_path=evidence_root["candidate_state"],
            candidate_frame_path=evidence_root["candidate_frame"],
            candidate_meta_path=evidence_root["candidate_meta"],
            protected_snapshot=evidence_root["snapshot"],
        )

    assert _destination_bytes(evidence_root) == before


def test_promote_rejects_tampered_candidate_png_without_touching_formal_pair(
    evidence_root: dict[str, Any],
) -> None:
    with evidence_root["candidate_frame"].open("ab") as output:
        output.write(b"tamper")
    before = _destination_bytes(evidence_root)

    with pytest.raises(ValueError, match=r"^frame\.annotated_sha256:"):
        promote_vision_evidence(
            root=evidence_root["root"],
            candidate_state_path=evidence_root["candidate_state"],
            candidate_frame_path=evidence_root["candidate_frame"],
            candidate_meta_path=evidence_root["candidate_meta"],
            protected_snapshot=evidence_root["snapshot"],
        )

    assert _destination_bytes(evidence_root) == before


def test_promote_rejects_candidate_video_path_incompatible_with_formal_state(
    evidence_root: dict[str, Any],
) -> None:
    candidate = json.loads(
        evidence_root["candidate_state"].read_text(encoding="utf-8")
    )
    candidate["source"]["video"] = "temporary/candidate.mp4"
    _write_json(evidence_root["candidate_state"], candidate)
    metadata = json.loads(
        evidence_root["candidate_meta"].read_text(encoding="utf-8")
    )
    metadata["video"]["path"] = "temporary/candidate.mp4"
    _write_json(evidence_root["candidate_meta"], metadata)
    before = _destination_bytes(evidence_root)

    with pytest.raises(ValueError, match=r"^video\.path:"):
        promote_vision_evidence(
            root=evidence_root["root"],
            candidate_state_path=evidence_root["candidate_state"],
            candidate_frame_path=evidence_root["candidate_frame"],
            candidate_meta_path=evidence_root["candidate_meta"],
            protected_snapshot=evidence_root["snapshot"],
        )

    assert _destination_bytes(evidence_root) == before


def test_promote_atomically_replaces_only_formal_evidence_pair(
    evidence_root: dict[str, Any],
) -> None:
    candidate_bytes = (
        evidence_root["candidate_frame"].read_bytes(),
        evidence_root["candidate_meta"].read_bytes(),
    )
    formal_state_bytes = evidence_root["formal_state"].read_bytes()

    result = promote_vision_evidence(
        root=evidence_root["root"],
        candidate_state_path=evidence_root["candidate_state"],
        candidate_frame_path=evidence_root["candidate_frame"],
        candidate_meta_path=evidence_root["candidate_meta"],
        protected_snapshot=evidence_root["snapshot"],
    )

    assert result == (
        evidence_root["destination_frame"],
        evidence_root["destination_meta"],
    )
    assert _destination_bytes(evidence_root) == candidate_bytes
    assert evidence_root["formal_state"].read_bytes() == formal_state_bytes
    verify_protected_data(evidence_root["root"], evidence_root["snapshot"])


def test_replace_pair_rolls_back_both_destinations_when_post_verify_fails(
    tmp_path: Path,
) -> None:
    source_frame = tmp_path / "source-frame"
    source_meta = tmp_path / "source-meta"
    destination_frame = tmp_path / "destination-frame"
    destination_meta = tmp_path / "destination-meta"
    source_frame.write_bytes(b"new-frame")
    source_meta.write_bytes(b"new-meta")
    destination_frame.write_bytes(b"old-frame")
    destination_meta.write_bytes(b"old-meta")

    def fail_verification() -> None:
        raise RuntimeError("post verification failed")

    with pytest.raises(RuntimeError, match="post verification failed"):
        _replace_pair_with_rollback(
            source_frame,
            source_meta,
            destination_frame,
            destination_meta,
            fail_verification,
        )

    assert destination_frame.read_bytes() == b"old-frame"
    assert destination_meta.read_bytes() == b"old-meta"


def test_replace_pair_stage_copy_failure_leaves_pair_and_no_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _raw_replacement_pair(tmp_path)
    real_copy2 = promotion_gate.shutil.copy2

    def fail_frame_stage_copy(source: Any, destination: Any) -> Any:
        if (
            Path(source) == paths["source_frame"]
            and Path(destination).suffix == ".tmp"
        ):
            raise OSError("stage copy failed")
        return real_copy2(source, destination)

    monkeypatch.setattr(promotion_gate.shutil, "copy2", fail_frame_stage_copy)

    with pytest.raises(OSError, match="stage copy failed"):
        _replace_pair_with_rollback(
            paths["source_frame"],
            paths["source_meta"],
            paths["destination_frame"],
            paths["destination_meta"],
            lambda: None,
        )

    assert _raw_destination_bytes(paths) == (b"old-frame", b"old-meta")
    assert _transaction_artifacts(tmp_path) == []


def test_replace_pair_first_replace_failure_leaves_pair_and_no_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _raw_replacement_pair(tmp_path)
    real_replace = promotion_gate.os.replace

    def fail_first_formal_replace(source: Any, destination: Any) -> Any:
        if (
            Path(source).suffix == ".tmp"
            and Path(destination) == paths["destination_frame"]
        ):
            raise OSError("first replace failed")
        return real_replace(source, destination)

    monkeypatch.setattr(promotion_gate.os, "replace", fail_first_formal_replace)

    with pytest.raises(OSError, match="first replace failed"):
        _replace_pair_with_rollback(
            paths["source_frame"],
            paths["source_meta"],
            paths["destination_frame"],
            paths["destination_meta"],
            lambda: None,
        )

    assert _raw_destination_bytes(paths) == (b"old-frame", b"old-meta")
    assert _transaction_artifacts(tmp_path) == []


def test_replace_pair_second_replace_failure_restores_pair_and_no_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _raw_replacement_pair(tmp_path)
    real_replace = promotion_gate.os.replace

    def fail_second_formal_replace(source: Any, destination: Any) -> Any:
        if (
            Path(source).suffix == ".tmp"
            and Path(destination) == paths["destination_meta"]
        ):
            raise OSError("second replace failed")
        return real_replace(source, destination)

    monkeypatch.setattr(promotion_gate.os, "replace", fail_second_formal_replace)

    with pytest.raises(OSError, match="second replace failed"):
        _replace_pair_with_rollback(
            paths["source_frame"],
            paths["source_meta"],
            paths["destination_frame"],
            paths["destination_meta"],
            lambda: None,
        )

    assert _raw_destination_bytes(paths) == (b"old-frame", b"old-meta")
    assert _transaction_artifacts(tmp_path) == []


def test_replace_pair_keeps_durable_backup_when_rollback_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _raw_replacement_pair(tmp_path)
    real_replace = promotion_gate.os.replace

    def fail_meta_replace_and_frame_rollback(source: Any, destination: Any) -> Any:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.suffix == ".tmp"
            and destination_path == paths["destination_meta"]
        ):
            raise OSError("second replace failed")
        if (
            source_path.suffix == ".backup"
            and destination_path == paths["destination_frame"]
        ):
            raise OSError("frame rollback failed")
        return real_replace(source, destination)

    monkeypatch.setattr(
        promotion_gate.os,
        "replace",
        fail_meta_replace_and_frame_rollback,
    )

    with pytest.raises(RuntimeError, match="rollback") as exc_info:
        _replace_pair_with_rollback(
            paths["source_frame"],
            paths["source_meta"],
            paths["destination_frame"],
            paths["destination_meta"],
            lambda: None,
        )

    frame_backups = list(
        tmp_path.glob(f".{paths['destination_frame'].name}.*.backup")
    )
    assert len(frame_backups) == 1
    assert frame_backups[0].read_bytes() == b"old-frame"
    error_message = str(exc_info.value)
    assert str(paths["destination_frame"].resolve()) in error_message
    assert str(frame_backups[0].resolve()) in error_message
    assert paths["destination_meta"].read_bytes() == b"old-meta"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_replace_pair_continues_other_rollback_after_first_restore_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _raw_replacement_pair(tmp_path)
    real_replace = promotion_gate.os.replace
    attempted_meta_rollback = False

    def fail_frame_rollback(source: Any, destination: Any) -> Any:
        nonlocal attempted_meta_rollback
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.suffix == ".backup"
            and destination_path == paths["destination_frame"]
        ):
            raise OSError("frame rollback failed")
        if (
            source_path.suffix == ".backup"
            and destination_path == paths["destination_meta"]
        ):
            attempted_meta_rollback = True
        return real_replace(source, destination)

    monkeypatch.setattr(promotion_gate.os, "replace", fail_frame_rollback)

    def fail_post_verification() -> None:
        raise ValueError("post verification failed")

    with pytest.raises(RuntimeError, match="rollback"):
        _replace_pair_with_rollback(
            paths["source_frame"],
            paths["source_meta"],
            paths["destination_frame"],
            paths["destination_meta"],
            fail_post_verification,
        )

    assert attempted_meta_rollback is True
    assert paths["destination_meta"].read_bytes() == b"old-meta"
    frame_backups = list(
        tmp_path.glob(f".{paths['destination_frame'].name}.*.backup")
    )
    assert len(frame_backups) == 1
    assert frame_backups[0].read_bytes() == b"old-frame"


def test_snapshot_and_verify_cli_round_trip(
    evidence_root: dict[str, Any], tmp_path: Path
) -> None:
    snapshot_path = tmp_path / "snapshots/protected.json"

    assert main(
        [
            "snapshot",
            "--root",
            str(evidence_root["root"]),
            "--out",
            str(snapshot_path),
        ]
    ) == 0
    assert json.loads(snapshot_path.read_text(encoding="utf-8")) == evidence_root[
        "snapshot"
    ]
    assert main(
        [
            "verify",
            "--root",
            str(evidence_root["root"]),
            "--snapshot",
            str(snapshot_path),
        ]
    ) == 0


def test_promote_cli_dispatches_real_promotion_and_preserves_snapshot(
    evidence_root: dict[str, Any], tmp_path: Path
) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    assert main(
        [
            "snapshot",
            "--root",
            str(evidence_root["root"]),
            "--out",
            str(snapshot_path),
        ]
    ) == 0
    snapshot_bytes = snapshot_path.read_bytes()
    candidate_bytes = (
        evidence_root["candidate_frame"].read_bytes(),
        evidence_root["candidate_meta"].read_bytes(),
    )

    assert main(
        [
            "promote",
            "--root",
            str(evidence_root["root"]),
            "--candidate-state",
            str(evidence_root["candidate_state"]),
            "--candidate-frame",
            str(evidence_root["candidate_frame"]),
            "--candidate-meta",
            str(evidence_root["candidate_meta"]),
            "--snapshot",
            str(snapshot_path),
        ]
    ) == 0

    assert _destination_bytes(evidence_root) == candidate_bytes
    assert snapshot_path.read_bytes() == snapshot_bytes
    verify_protected_data(evidence_root["root"], evidence_root["snapshot"])

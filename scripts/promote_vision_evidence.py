from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vision.frame_evidence import load_and_validate_frame_evidence, sha256_file


PROTECTED_FILES = (
    "data/traffic_states/demo_001.json",
    "data/results/arena_summary.csv",
)
PROTECTED_TREES = ("data/results/experiments",)


def traffic_state_semantic_view(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": state["schema_version"],
        "duration_sec": state["duration_sec"],
        "approaches": {
            "north": {
                "observed": state["approaches"]["north"]["observed"],
                "flow_vph": state["approaches"]["north"]["flow_vph"],
                "queue_est": state["approaches"]["north"]["queue_est"],
                "vehicle_mix": state["approaches"]["north"]["vehicle_mix"],
            },
            "south": {
                "observed": state["approaches"]["south"]["observed"],
                "flow_vph": state["approaches"]["south"]["flow_vph"],
                "queue_est": state["approaches"]["south"]["queue_est"],
                "vehicle_mix": state["approaches"]["south"]["vehicle_mix"],
            },
            "east": {
                "observed": state["approaches"]["east"]["observed"],
                "flow_vph": state["approaches"]["east"]["flow_vph"],
                "queue_est": state["approaches"]["east"]["queue_est"],
                "vehicle_mix": state["approaches"]["east"]["vehicle_mix"],
            },
            "west": {
                "observed": state["approaches"]["west"]["observed"],
                "flow_vph": state["approaches"]["west"]["flow_vph"],
                "queue_est": state["approaches"]["west"]["queue_est"],
                "vehicle_mix": state["approaches"]["west"]["vehicle_mix"],
            },
        },
        "turning_ratio": state["turning_ratio"],
        "flow_profile": state["flow_profile"],
        "profile_bins_sec": state["profile_bins_sec"],
        "source": {
            "fps": state["source"]["fps"],
            "frames": state["source"]["frames"],
            "duration_sec": state["source"]["duration_sec"],
        },
    }


def _path_child(prefix: str, key: object) -> str:
    return f"{prefix}.{key}" if prefix else str(key)


def _dotted_differences(
    reference: Any,
    candidate: Any,
    prefix: str = "",
) -> list[str]:
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        differences: list[str] = []
        for key in sorted(set(reference) | set(candidate), key=str):
            path = _path_child(prefix, key)
            if key not in reference or key not in candidate:
                differences.append(path)
                continue
            differences.extend(
                _dotted_differences(reference[key], candidate[key], path)
            )
        return differences

    reference_is_sequence = isinstance(reference, Sequence) and not isinstance(
        reference, (str, bytes, bytearray)
    )
    candidate_is_sequence = isinstance(candidate, Sequence) and not isinstance(
        candidate, (str, bytes, bytearray)
    )
    if reference_is_sequence and candidate_is_sequence:
        if len(reference) != len(candidate):
            return [prefix]
        differences = []
        for index, (reference_item, candidate_item) in enumerate(
            zip(reference, candidate, strict=True)
        ):
            differences.extend(
                _dotted_differences(
                    reference_item,
                    candidate_item,
                    f"{prefix}[{index}]",
                )
            )
        return differences

    if reference != candidate:
        return [prefix]
    return []


def semantic_differences(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[str]:
    return _dotted_differences(
        traffic_state_semantic_view(reference),
        traffic_state_semantic_view(candidate),
    )


def sha256_tree(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    files = sorted(
        (
            (file_path.relative_to(path).as_posix(), file_path)
            for file_path in path.rglob("*")
            if file_path.is_file()
        ),
        key=lambda item: item[0],
    )
    for relative_path, file_path in files:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def snapshot_protected_data(root: Path) -> dict[str, dict[str, str]]:
    root = Path(root).resolve()
    return {
        "files": {
            relative_path: sha256_file(root / relative_path)
            for relative_path in PROTECTED_FILES
        },
        "trees": {
            relative_path: sha256_tree(root / relative_path)
            for relative_path in PROTECTED_TREES
        },
    }


def verify_protected_data(
    root: Path,
    expected: Mapping[str, Any],
) -> None:
    actual = snapshot_protected_data(Path(root).resolve())
    changed: list[str] = []
    expected_group_names = set(expected)
    actual_group_names = set(actual)
    changed.extend(
        str(group)
        for group in sorted(expected_group_names ^ actual_group_names, key=str)
    )

    for group_name in sorted(actual_group_names):
        expected_group = expected.get(group_name)
        actual_group = actual[group_name]
        if not isinstance(expected_group, Mapping):
            changed.extend(actual_group)
            continue
        for relative_path in sorted(set(expected_group) | set(actual_group), key=str):
            if (
                relative_path not in expected_group
                or relative_path not in actual_group
                or expected_group[relative_path] != actual_group[relative_path]
            ):
                changed.append(str(relative_path))

    if changed:
        raise ValueError(
            "protected data changed: " + ", ".join(sorted(set(changed)))
        )


def _replace_pair_with_rollback(
    source_frame: str | Path,
    source_meta: str | Path,
    destination_frame: str | Path,
    destination_meta: str | Path,
    verify_after_replace: Callable[[], Any],
) -> None:
    source_frame = Path(source_frame)
    source_meta = Path(source_meta)
    destination_frame = Path(destination_frame)
    destination_meta = Path(destination_meta)
    destination_frame.parent.mkdir(parents=True, exist_ok=True)
    destination_meta.parent.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex
    staged_frame = destination_frame.parent / (
        f".{destination_frame.name}.{token}.tmp"
    )
    staged_meta = destination_meta.parent / f".{destination_meta.name}.{token}.tmp"
    backup_frame = destination_frame.parent / (
        f".{destination_frame.name}.{token}.backup"
    )
    backup_meta = destination_meta.parent / (
        f".{destination_meta.name}.{token}.backup"
    )
    frame_existed = destination_frame.exists()
    meta_existed = destination_meta.exists()
    frame_replaced = False
    meta_replaced = False
    preserve_backups = False

    try:
        if frame_existed:
            shutil.copy2(destination_frame, backup_frame)
        if meta_existed:
            shutil.copy2(destination_meta, backup_meta)
        shutil.copy2(source_frame, staged_frame)
        shutil.copy2(source_meta, staged_meta)
        os.replace(staged_frame, destination_frame)
        frame_replaced = True
        os.replace(staged_meta, destination_meta)
        meta_replaced = True
        verify_after_replace()
    except Exception as original_error:
        rollback_errors: list[tuple[Path, Exception]] = []
        rollback_targets = (
            (
                destination_frame,
                frame_replaced,
                backup_frame if frame_existed else None,
            ),
            (
                destination_meta,
                meta_replaced,
                backup_meta if meta_existed else None,
            ),
        )
        for destination, replaced, backup in rollback_targets:
            if not replaced:
                continue
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            except Exception as rollback_error:
                rollback_errors.append((destination, rollback_error))

        if rollback_errors:
            preserve_backups = True
            failed_targets = ", ".join(
                f"{destination.resolve()} ({error})"
                for destination, error in rollback_errors
            )
            retained_backups = [
                str(backup.resolve())
                for backup in (backup_frame, backup_meta)
                if backup.exists()
            ]
            retained_description = (
                ", ".join(retained_backups) if retained_backups else "<none>"
            )
            raise RuntimeError(
                f"rollback failed for {failed_targets}; "
                f"retained backups: {retained_description}"
            ) from original_error
        raise
    finally:
        staged_frame.unlink(missing_ok=True)
        staged_meta.unlink(missing_ok=True)
        if not preserve_backups:
            backup_frame.unlink(missing_ok=True)
            backup_meta.unlink(missing_ok=True)


def promote_vision_evidence(
    *,
    root: Path,
    candidate_state_path: Path,
    candidate_frame_path: Path,
    candidate_meta_path: Path,
    protected_snapshot: Mapping[str, Any],
) -> tuple[Path, Path]:
    root = Path(root).resolve()
    reference_state_path = root / "data/traffic_states/demo_001.json"
    reference_state = json.loads(reference_state_path.read_text(encoding="utf-8"))
    candidate_state = json.loads(
        Path(candidate_state_path).read_text(encoding="utf-8")
    )

    differences = semantic_differences(reference_state, candidate_state)
    if differences:
        raise ValueError(
            "TrafficState semantic mismatch: " + ", ".join(differences)
        )

    verify_protected_data(root, protected_snapshot)

    source_video = str(reference_state["source"]["video"]).replace("\\", "/")
    video_path = root / source_video
    roi_path = root / "data/videos/demo_roi.json"
    candidate_frame_path = Path(candidate_frame_path)
    candidate_meta_path = Path(candidate_meta_path)
    load_and_validate_frame_evidence(
        candidate_frame_path,
        candidate_meta_path,
        video_path,
        roi_path,
        reference_state,
    )

    destination_frame = root / "figures/fig_vision_annotated_frame.png"
    destination_meta = root / "figures/fig_vision_annotated_frame.meta.json"

    def verify_after_replace() -> None:
        load_and_validate_frame_evidence(
            destination_frame,
            destination_meta,
            video_path,
            roi_path,
            reference_state,
        )
        verify_protected_data(root, protected_snapshot)

    _replace_pair_with_rollback(
        candidate_frame_path,
        candidate_meta_path,
        destination_frame,
        destination_meta,
        verify_after_replace,
    )
    return destination_frame, destination_meta


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Protect and promote vision evidence"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--root", type=Path, required=True)
    snapshot_parser.add_argument("--out", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--snapshot", type=Path, required=True)

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--root", type=Path, required=True)
    promote_parser.add_argument("--candidate-state", type=Path, required=True)
    promote_parser.add_argument("--candidate-frame", type=Path, required=True)
    promote_parser.add_argument("--candidate-meta", type=Path, required=True)
    promote_parser.add_argument("--snapshot", type=Path, required=True)
    return parser


def _read_snapshot(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "snapshot":
        snapshot = snapshot_protected_data(root)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    if args.command == "verify":
        verify_protected_data(root, _read_snapshot(args.snapshot))
        return 0
    if args.command == "promote":
        promote_vision_evidence(
            root=root,
            candidate_state_path=args.candidate_state,
            candidate_frame_path=args.candidate_frame,
            candidate_meta_path=args.candidate_meta,
            protected_snapshot=_read_snapshot(args.snapshot),
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

import numpy as np

from vision.frame_evidence import RepresentativeFrameSelector, TrackEvidence


def _image(value: int) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


def _track(track_id: int, trace_length: int) -> TrackEvidence:
    return TrackEvidence(
        track_id=track_id,
        class_name="car",
        bbox_xyxy=(5.0, 5.0, 15.0, 15.0),
        trace_points=tuple((float(i), float(i)) for i in range(trace_length)),
    )


def test_active_track_count_takes_priority_over_trace_point_count() -> None:
    selector = RepresentativeFrameSelector(fps=30.0)
    selector.consider(
        annotated_bgr=_image(0),
        frame_index=0,
        tracks=(_track(1, 60),),
    )
    selector.consider(
        annotated_bgr=_image(1),
        frame_index=1,
        tracks=(_track(1, 1), _track(2, 1)),
    )

    result = selector.result()

    assert result.frame_index == 1
    assert result.active_track_count == 2
    assert result.active_trace_point_count == 2


def test_trace_point_count_breaks_ties_between_equal_track_counts() -> None:
    selector = RepresentativeFrameSelector(fps=25.0)
    selector.consider(
        annotated_bgr=_image(0),
        frame_index=0,
        tracks=(_track(1, 2), _track(2, 2)),
    )
    selector.consider(
        annotated_bgr=_image(1),
        frame_index=1,
        tracks=(_track(1, 3), _track(2, 2)),
    )

    result = selector.result()

    assert result.frame_index == 1
    assert result.active_track_count == 2
    assert result.active_trace_point_count == 5


def test_earlier_frame_breaks_full_tie_and_owns_an_image_copy() -> None:
    selector = RepresentativeFrameSelector(fps=20.0)
    selector.consider(
        annotated_bgr=_image(9),
        frame_index=4,
        tracks=(_track(1, 2), _track(2, 1)),
    )
    earlier_image = _image(3)
    selector.consider(
        annotated_bgr=earlier_image,
        frame_index=2,
        tracks=(_track(1, 2), _track(2, 1)),
    )
    earlier_image[:] = 99

    result = selector.result()

    assert result.frame_index == 2
    assert result.timestamp_sec == 0.1
    assert result.score == (2, 3, -2)
    np.testing.assert_array_equal(result.annotated_bgr, _image(3))


def test_no_detections_returns_first_fully_annotated_frame() -> None:
    selector = RepresentativeFrameSelector(fps=10.0)
    first_image = _image(7)
    selector.consider(annotated_bgr=first_image, frame_index=0, tracks=())
    selector.consider(annotated_bgr=_image(8), frame_index=1, tracks=())

    result = selector.result()

    assert result.frame_index == 0
    assert result.no_detections is True
    np.testing.assert_array_equal(result.annotated_bgr, first_image)

from __future__ import annotations

import pytest

from racing.track.world import (
    MUGELLO_SHORT_LAYOUT,
    TRACK_ID_MUGELLO_SHORT,
    sampled_track_centerline,
    total_track_length,
    track_bounds,
    track_layout_by_id,
    track_layout_ids,
)


def test_default_track_layout_is_available() -> None:
    assert TRACK_ID_MUGELLO_SHORT in track_layout_ids()
    assert track_layout_by_id(TRACK_ID_MUGELLO_SHORT).track_id == TRACK_ID_MUGELLO_SHORT


def test_unknown_track_layout_reports_valid_ids() -> None:
    with pytest.raises(ValueError, match=TRACK_ID_MUGELLO_SHORT):
        track_layout_by_id("missing")


def test_track_length_and_bounds_are_positive() -> None:
    bounds = track_bounds(MUGELLO_SHORT_LAYOUT)

    assert total_track_length(MUGELLO_SHORT_LAYOUT) > 0.0
    assert bounds.width > 0.0
    assert bounds.length > 0.0


def test_sampled_track_centerline_requires_positive_samples() -> None:
    with pytest.raises(ValueError, match="samples_per_segment"):
        sampled_track_centerline(MUGELLO_SHORT_LAYOUT, samples_per_segment=0)

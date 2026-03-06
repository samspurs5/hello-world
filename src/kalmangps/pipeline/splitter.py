"""Trajectory splitting logic.

A raw GPS log often contains multiple trips / devices / sessions mixed
together.  Before filtering we need to:

1. **Split by a user column** (e.g. ``device_id``, ``trip_id``, ``user_id``).
2. **Split each group by time gaps** — a long pause between consecutive fixes
   means the device was stationary or off; treating it as one continuous
   trajectory would let the Kalman filter interpolate through the gap rather
   than treating the restart as a fresh segment.

The output of both stages is a list of ``TrajectorySegment`` objects — plain
dataclasses that bundle the subset DataFrame with its identifying metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class TrajectorySegment:
    """One continuous, homogeneous slice of a GPS trajectory.

    Attributes
    ----------
    data:
        The DataFrame rows belonging to this segment, sorted by timestamp.
    group_key:
        Value of the user-supplied split column for this group (or ``None``
        when no column split was requested).
    segment_index:
        Zero-based index of this segment *within* its group (after time-gap
        splitting).  Useful for debugging and output labelling.
    metadata:
        Arbitrary extra information (e.g. device type, route name).
    """

    data: pd.DataFrame
    group_key: Any = None
    segment_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"TrajectorySegment(group_key={self.group_key!r}, "
            f"segment_index={self.segment_index}, n_points={self.n_points})"
        )


class TrajectorySplitter:
    """Split a GPS DataFrame into filtered-ready ``TrajectorySegment`` objects.

    Parameters
    ----------
    group_col:
        Column name to group by before time splitting (e.g. ``"trip_id"``).
        Pass ``None`` to treat the whole DataFrame as a single group.
    timestamp_col:
        Column name for the UTC timestamp used for time-gap splitting and
        for sorting within each group.
    max_time_gap:
        Maximum allowed gap between consecutive observations.  When two
        consecutive rows are further apart than this, a new segment begins.
        Accepts a ``pd.Timedelta`` or anything that ``pd.Timedelta`` accepts
        (e.g. ``"5 min"``, ``300`` seconds as int).
    min_segment_length:
        Segments with fewer observations than this are silently dropped —
        they don't carry enough information for the Kalman filter to be
        meaningful.
    """

    def __init__(
        self,
        group_col: str | None = None,
        timestamp_col: str = "timestamp",
        max_time_gap: pd.Timedelta | str | int = pd.Timedelta("5 min"),
        min_segment_length: int = 3,
    ) -> None:
        self.group_col = group_col
        self.timestamp_col = timestamp_col
        self.max_time_gap = pd.Timedelta(max_time_gap) if not isinstance(
            max_time_gap, pd.Timedelta
        ) else max_time_gap
        self.min_segment_length = min_segment_length

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def split(self, df: pd.DataFrame) -> list[TrajectorySegment]:
        """Split *df* into trajectory segments.

        Parameters
        ----------
        df:
            Input DataFrame.  Must contain ``timestamp_col`` and, when
            ``group_col`` is set, ``group_col``.

        Returns
        -------
        List of :class:`TrajectorySegment` objects ready for filtering.
        """
        if self.timestamp_col not in df.columns:
            raise ValueError(
                f"timestamp column '{self.timestamp_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], utc=True)

        if self.group_col is not None:
            if self.group_col not in df.columns:
                raise ValueError(
                    f"group column '{self.group_col}' not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )
            groups = df.groupby(self.group_col, sort=False)
        else:
            groups = [(None, df)]

        segments: list[TrajectorySegment] = []
        for key, group_df in groups:
            group_segs = self._split_by_time(group_df, group_key=key)
            segments.extend(group_segs)

        return segments

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _split_by_time(
        self, df: pd.DataFrame, group_key: Any
    ) -> list[TrajectorySegment]:
        """Split a single-group DataFrame on time gaps."""
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)
        times = df[self.timestamp_col]

        # Compute boolean mask: True where a new segment starts
        diffs = times.diff()
        new_segment = (diffs > self.max_time_gap) | (diffs.isna())

        segment_id = new_segment.cumsum() - 1  # 0-based

        segments: list[TrajectorySegment] = []
        for seg_idx, seg_df in df.groupby(segment_id):
            if len(seg_df) < self.min_segment_length:
                continue
            segments.append(
                TrajectorySegment(
                    data=seg_df.reset_index(drop=True),
                    group_key=group_key,
                    segment_index=int(seg_idx),
                )
            )

        return segments

"""Tests for integer 3D label volume export and visual inspection fidelity."""

from __future__ import annotations

import numpy as np
import tifffile

from nervenuc.fiji.exchange import FijiExchangeSession


def test_write_label_volume_preserves_integer_ids() -> None:
    """Verify integer label IDs are preserved with no normalization or interpolation."""
    # Discrete label IDs
    labels = np.zeros((4, 15, 15), dtype=np.int32)
    labels[0:2, 2:8, 2:8] = 101
    labels[1:3, 8:14, 8:14] = 202
    labels[2:4, 3:7, 9:13] = 303

    with FijiExchangeSession() as session:
        p = session.write_exchange_tiff(labels, "ZYX", is_labels=True)

        read_back = tifffile.imread(str(p))
        assert read_back.shape == (4, 15, 15)
        # Verify unique label IDs are unchanged
        unique_orig = set(np.unique(labels))
        unique_read = set(np.unique(read_back))
        assert unique_orig == unique_read == {0, 101, 202, 303}
        assert np.array_equal(labels, read_back)

"""Differential Index — autonomic marker: nose tip temperature minus periorbital temperature."""


def differential_index(roi_temps: dict) -> float:
    """roi_temps: {roi_name: mean_temp_c}, as returned by src.roi.extraction.extract_roi_temperatures."""
    return roi_temps["nose_tip"] - roi_temps["periorbital"]

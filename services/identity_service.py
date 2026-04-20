"""
Preparation layer for future employee identification.

This module intentionally does not implement full face recognition yet.
It defines stable extension points so the current monitoring system can
keep working without identification being configured.
"""

from typing import Optional


def get_identity_placeholder_result() -> dict:
    """Return a safe default when identity recognition is not configured."""
    return {
        "identified_employee_id": None,
        "identification_confidence": None,
        "identification_status": "not_configured",
    }


def detect_face(frame_bgr, detection: Optional[dict] = None):
    """
    Placeholder for future face detection inside a person region.

    Expected future behavior:
    - receive a frame and optional person detection metadata;
    - return a cropped face region or structured face detection result.
    """
    return None


def extract_embedding(face_image):
    """
    Placeholder for future facial embedding extraction.

    Expected future behavior:
    - receive a cropped face image;
    - return a numeric embedding vector for employee matching.
    """
    return None


def match_employee(embedding, employee_gallery=None):
    """
    Placeholder for future employee matching against a reference gallery.

    Expected future behavior:
    - compare an embedding with stored employee references;
    - return matched employee id and confidence score.
    """
    return get_identity_placeholder_result()


def identify_person(frame_bgr=None, detection: Optional[dict] = None, employee_gallery=None):
    """
    High-level placeholder for employee identification.

    The current system keeps working as person monitoring only. When the
    identification stack is added, this function can orchestrate:
    face detection -> embedding extraction -> employee matching.
    """
    _ = frame_bgr
    _ = detection
    _ = employee_gallery
    return get_identity_placeholder_result()

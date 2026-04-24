"""
Preparation layer for future employee verification.

The system must not invent identification results when reference data or
matching infrastructure are missing. This module therefore exposes stable
extension points and returns explicit production-safe statuses.
"""

from __future__ import annotations

from typing import Optional

from config.app_config import build_identity_backend_config


IDENTIFICATION_STATUSES = {
    "no_reference_data",
    "not_enough_reference_data",
    "db_unavailable",
    "low_confidence",
    "inactive_employee",
    "unknown",
    "unlinked",
    "pending_operator_confirmation",
    "linked_from_directory",
    "linked_from_access_control",
}


def get_identity_placeholder_result(identity_state: Optional[dict] = None) -> dict:
    """Return a safe default identity result based on directory health and reference data."""
    status = resolve_identification_status(identity_state)
    return {
        "identified_employee_id": None,
        "identification_confidence": None,
        "identification_status": status,
    }


def get_disabled_identity_result() -> dict:
    return {
        "identified_employee_id": None,
        "identification_confidence": None,
        "identification_status": "unlinked",
    }


def resolve_identification_status(identity_state: Optional[dict] = None) -> str:
    """Map the current employee directory state to an honest identification status."""
    if not identity_state:
        return "unlinked"

    sync_status = identity_state.get("sync_status") or "unknown"
    sync_error = (identity_state.get("sync_error") or "").strip()
    employee_count = int(identity_state.get("employee_count") or 0)
    active_employee_count = int(identity_state.get("active_employee_count") or 0)
    reference_employee_count = int(identity_state.get("reference_employee_count") or 0)

    if sync_status == "fallback_cache" and sync_error:
        return "db_unavailable"
    if employee_count == 0 or active_employee_count == 0:
        return "no_reference_data"
    if reference_employee_count == 0:
        return "no_reference_data"
    if reference_employee_count < min(active_employee_count, 3):
        return "not_enough_reference_data"
    return "unknown"


def resolve_identity_backend(identity_state: Optional[dict] = None) -> dict:
    identity_state = identity_state or {}
    return build_identity_backend_config(identity_state.get("identity_backend"))


def build_identity_result(
    *,
    identified_employee: Optional[dict] = None,
    confidence: Optional[float] = None,
    identity_state: Optional[dict] = None,
    ambiguous: bool = False,
) -> dict:
    """Build a normalized identity result for event journaling."""
    if ambiguous:
        return {
            "identified_employee_id": None,
            "identification_confidence": confidence,
            "identification_status": "pending_operator_confirmation",
        }
    if identified_employee is None:
        return get_identity_placeholder_result(identity_state)
    if identified_employee.get("status") != "active":
        return {
            "identified_employee_id": identified_employee.get("id"),
            "identification_confidence": confidence,
            "identification_status": "inactive_employee",
        }
    if confidence is None:
        return {
            "identified_employee_id": identified_employee.get("id"),
            "identification_confidence": None,
            "identification_status": "linked_from_directory",
        }
    if confidence < 0.65:
        return {
            "identified_employee_id": identified_employee.get("id"),
            "identification_confidence": confidence,
            "identification_status": "low_confidence",
        }
    return {
        "identified_employee_id": identified_employee.get("id"),
        "identification_confidence": confidence,
        "identification_status": "linked_from_directory",
    }


def detect_face(frame_bgr, detection: Optional[dict] = None):
    """
    Placeholder for future face detection inside a person region.

    Expected future behavior:
    - receive a frame and optional person detection metadata;
    - return a cropped face region or structured face detection result.
    """
    _ = frame_bgr
    _ = detection
    return None


def extract_embedding(face_image):
    """
    Placeholder for future facial embedding extraction.

    Expected future behavior:
    - receive a cropped face image;
    - return a numeric embedding vector for employee matching.
    """
    _ = face_image
    return None


def match_employee(embedding, employee_gallery=None):
    """
    Placeholder for future employee matching against a reference gallery.

    Current behavior remains intentionally conservative: without a real
    embedding gallery and similarity thresholds, no employee match is inferred.
    """
    _ = embedding
    _ = employee_gallery
    return None


def identify_person(
    frame_bgr=None,
    detection: Optional[dict] = None,
    employee_gallery=None,
    identity_state: Optional[dict] = None,
):
    """
    High-level identification orchestration with graceful degradation.

    Current pipeline:
    - evaluate availability of employee directory and reference data;
    - if data is insufficient, return explicit status;
    - if future face recognition pieces are configured, this function becomes
      the single orchestration entry point.
    """
    backend = resolve_identity_backend(identity_state)
    if not backend["enabled"]:
        return get_disabled_identity_result()

    face = detect_face(frame_bgr, detection=detection)
    if face is None:
        return get_identity_placeholder_result(identity_state)

    embedding = extract_embedding(face)
    if embedding is None:
        return get_identity_placeholder_result(identity_state)

    matched_employee = match_employee(embedding, employee_gallery=employee_gallery)
    if matched_employee is None:
        return get_identity_placeholder_result(identity_state)
    return build_identity_result(
        identified_employee=matched_employee,
        confidence=matched_employee.get("confidence"),
        identity_state=identity_state,
    )


def build_identity_runtime_state(
    *,
    employees: list[dict],
    sync_state: dict | None,
    identity_backend: str,
) -> dict:
    sync_state = sync_state or {}
    backend = build_identity_backend_config(identity_backend)
    active_employees = [employee for employee in employees if employee.get("status") == "active"]
    referenced_employees = [employee for employee in active_employees if int(employee.get("reference_count") or 0) > 0]
    return {
        "employee_count": len(employees),
        "active_employee_count": len(active_employees),
        "reference_employee_count": len(referenced_employees),
        "directory_source": sync_state.get("data_source", "sqlite"),
        "sync_status": sync_state.get("sync_status", "unknown"),
        "sync_error": sync_state.get("last_error", ""),
        "identity_backend": backend["backend"],
        "identity_backend_label": backend["label"],
        "identity_backend_enabled": backend["enabled"],
    }

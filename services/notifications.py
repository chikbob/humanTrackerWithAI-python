"""Incident notification delivery helpers."""

from __future__ import annotations

import json
from urllib import error, parse, request


SEVERITY_PRIORITY = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


def process_incident_notifications(
    *,
    incidents: list[dict],
    settings: dict,
    load_notification_deliveries_fn,
    upsert_notification_delivery_fn,
    webhook_sender=None,
    telegram_sender=None,
):
    if str(settings.get("notifications_enabled", "0")) != "1":
        return []

    deliveries = load_notification_deliveries_fn()
    sent_keys = {
        (delivery["incident_id"], delivery["channel"], delivery["destination"])
        for delivery in deliveries
        if delivery.get("delivery_status") == "sent"
    }
    min_severity = settings.get("incident_notify_min_severity", "high")
    results = []
    for incident in incidents:
        if not _should_notify_incident(incident, min_severity=min_severity):
            continue
        results.extend(
            _deliver_incident(
                incident=incident,
                settings=settings,
                sent_keys=sent_keys,
                upsert_notification_delivery_fn=upsert_notification_delivery_fn,
                webhook_sender=webhook_sender or send_webhook_notification,
                telegram_sender=telegram_sender or send_telegram_notification,
            )
        )
    return results


def _deliver_incident(
    *,
    incident: dict,
    settings: dict,
    sent_keys: set[tuple[int, str, str]],
    upsert_notification_delivery_fn,
    webhook_sender,
    telegram_sender,
):
    results = []
    webhook_url = (settings.get("webhook_url") or "").strip()
    if str(settings.get("webhook_enabled", "0")) == "1" and webhook_url:
        key = (incident["id"], "webhook", webhook_url)
        if key not in sent_keys:
            try:
                webhook_sender(webhook_url, _build_incident_payload(incident))
            except Exception as exc:
                upsert_notification_delivery_fn(
                    incident_id=incident["id"],
                    channel="webhook",
                    destination=webhook_url,
                    delivery_status="failed",
                    last_error=f"{type(exc).__name__}: {exc}",
                )
                results.append({"incident_id": incident["id"], "channel": "webhook", "status": "failed"})
            else:
                upsert_notification_delivery_fn(
                    incident_id=incident["id"],
                    channel="webhook",
                    destination=webhook_url,
                    delivery_status="sent",
                    sent_at=incident.get("updated_at"),
                )
                sent_keys.add(key)
                results.append({"incident_id": incident["id"], "channel": "webhook", "status": "sent"})

    bot_token = (settings.get("telegram_bot_token") or "").strip()
    chat_id = (settings.get("telegram_chat_id") or "").strip()
    if str(settings.get("telegram_enabled", "0")) == "1" and bot_token and chat_id:
        destination = f"{chat_id}"
        key = (incident["id"], "telegram", destination)
        if key not in sent_keys:
            try:
                telegram_sender(bot_token, chat_id, build_telegram_message(incident))
            except Exception as exc:
                upsert_notification_delivery_fn(
                    incident_id=incident["id"],
                    channel="telegram",
                    destination=destination,
                    delivery_status="failed",
                    last_error=f"{type(exc).__name__}: {exc}",
                )
                results.append({"incident_id": incident["id"], "channel": "telegram", "status": "failed"})
            else:
                upsert_notification_delivery_fn(
                    incident_id=incident["id"],
                    channel="telegram",
                    destination=destination,
                    delivery_status="sent",
                    sent_at=incident.get("updated_at"),
                )
                sent_keys.add(key)
                results.append({"incident_id": incident["id"], "channel": "telegram", "status": "sent"})
    return results


def _should_notify_incident(incident: dict, *, min_severity: str) -> bool:
    if incident.get("status") not in {"new", "escalated"}:
        return False
    incident_priority = SEVERITY_PRIORITY.get(incident.get("severity"), 0)
    min_priority = SEVERITY_PRIORITY.get(min_severity, 2)
    return incident_priority >= min_priority


def _build_incident_payload(incident: dict) -> dict:
    return {
        "incident_id": incident["id"],
        "event_id": incident.get("event_id"),
        "type": incident.get("incident_type"),
        "severity": incident.get("severity"),
        "status": incident.get("status"),
        "source_id": incident.get("source_id"),
        "source_name": incident.get("source_name"),
        "zone_name": incident.get("zone_name"),
        "confidence": incident.get("confidence"),
        "snapshot_path": incident.get("snapshot_path"),
        "started_at": incident.get("started_at"),
        "updated_at": incident.get("updated_at"),
    }


def build_telegram_message(incident: dict) -> str:
    return (
        f"[{incident.get('severity', 'medium').upper()}] {incident.get('incident_type', 'incident')}\n"
        f"Источник: {incident.get('source_name') or incident.get('source_id') or '—'}\n"
        f"Зона: {incident.get('zone_name') or 'не задана'}\n"
        f"Статус: {incident.get('status') or 'new'}\n"
        f"Confidence: {round(float(incident.get('confidence') or 0.0), 3)}"
    )


def send_webhook_notification(url: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=5) as response:
        status = getattr(response, "status", None) or response.getcode()
        if status >= 400:
            raise RuntimeError(f"webhook_http_{status}")


def send_telegram_notification(bot_token: str, chat_id: str, text: str):
    endpoint = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    body = parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = request.Request(endpoint, data=body, method="POST")
    try:
        with request.urlopen(req, timeout=5) as response:
            status = getattr(response, "status", None) or response.getcode()
            if status >= 400:
                raise RuntimeError(f"telegram_http_{status}")
    except error.HTTPError as exc:
        raise RuntimeError(f"telegram_http_{exc.code}") from exc

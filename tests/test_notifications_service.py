import unittest

from services.notifications import build_telegram_message, process_incident_notifications


class NotificationsServiceTests(unittest.TestCase):
    def test_sent_delivery_is_not_repeated_for_same_channel(self):
        calls = []
        deliveries = [
            {
                "incident_id": 11,
                "channel": "webhook",
                "destination": "https://example.test/hook",
                "delivery_status": "sent",
            }
        ]

        results = process_incident_notifications(
            incidents=[
                {
                    "id": 11,
                    "severity": "critical",
                    "status": "new",
                    "incident_type": "intrusion",
                    "updated_at": "2026-04-24T10:00:00",
                }
            ],
            settings={
                "notifications_enabled": "1",
                "incident_notify_min_severity": "high",
                "webhook_enabled": "1",
                "webhook_url": "https://example.test/hook",
            },
            load_notification_deliveries_fn=lambda: deliveries,
            upsert_notification_delivery_fn=lambda **kwargs: calls.append(kwargs),
            webhook_sender=lambda *_args, **_kwargs: self.fail("webhook sender should not be called"),
        )

        self.assertEqual(results, [])
        self.assertEqual(calls, [])

    def test_threshold_and_status_filtering(self):
        sent = []
        incidents = [
            {"id": 1, "severity": "medium", "status": "new", "incident_type": "presence"},
            {"id": 2, "severity": "critical", "status": "resolved", "incident_type": "intrusion"},
            {"id": 3, "severity": "high", "status": "escalated", "incident_type": "intrusion"},
        ]

        process_incident_notifications(
            incidents=incidents,
            settings={
                "notifications_enabled": "1",
                "incident_notify_min_severity": "high",
                "webhook_enabled": "1",
                "webhook_url": "https://example.test/hook",
            },
            load_notification_deliveries_fn=lambda: [],
            upsert_notification_delivery_fn=lambda **kwargs: sent.append(kwargs),
            webhook_sender=lambda *_args, **_kwargs: None,
        )

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["incident_id"], 3)
        self.assertEqual(sent[0]["delivery_status"], "sent")

    def test_failed_delivery_is_persisted(self):
        deliveries = []

        process_incident_notifications(
            incidents=[
                {"id": 21, "severity": "high", "status": "new", "incident_type": "loitering"},
            ],
            settings={
                "notifications_enabled": "1",
                "incident_notify_min_severity": "high",
                "webhook_enabled": "1",
                "webhook_url": "https://example.test/hook",
            },
            load_notification_deliveries_fn=lambda: [],
            upsert_notification_delivery_fn=lambda **kwargs: deliveries.append(kwargs),
            webhook_sender=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("network_down")),
        )

        self.assertEqual(len(deliveries), 1)
        self.assertEqual(deliveries[0]["incident_id"], 21)
        self.assertEqual(deliveries[0]["delivery_status"], "failed")
        self.assertIn("network_down", deliveries[0]["last_error"])

    def test_telegram_delivery_is_recorded(self):
        deliveries = []

        process_incident_notifications(
            incidents=[
                {
                    "id": 31,
                    "severity": "critical",
                    "status": "new",
                    "incident_type": "camera_offline",
                    "source_name": "Склад 1",
                    "zone_name": "Периметр",
                    "confidence": 0.99,
                    "updated_at": "2026-04-24T11:00:00",
                },
            ],
            settings={
                "notifications_enabled": "1",
                "incident_notify_min_severity": "high",
                "telegram_enabled": "1",
                "telegram_bot_token": "token",
                "telegram_chat_id": "-100111",
            },
            load_notification_deliveries_fn=lambda: [],
            upsert_notification_delivery_fn=lambda **kwargs: deliveries.append(kwargs),
            telegram_sender=lambda bot_token, chat_id, text: self.assertIn("camera_offline", text),
        )

        self.assertEqual(len(deliveries), 1)
        self.assertEqual(deliveries[0]["channel"], "telegram")
        self.assertEqual(deliveries[0]["destination"], "-100111")
        self.assertEqual(deliveries[0]["delivery_status"], "sent")

    def test_build_telegram_message_contains_context(self):
        text = build_telegram_message(
            {
                "severity": "high",
                "incident_type": "intrusion",
                "source_name": "Камера 2",
                "zone_name": "Вход",
                "status": "new",
                "confidence": 0.8123,
            }
        )
        self.assertIn("[HIGH]", text)
        self.assertIn("Камера 2", text)
        self.assertIn("Вход", text)
        self.assertIn("0.812", text)


if __name__ == "__main__":
    unittest.main()

import unittest

from services.auth import get_visible_sections, has_permission


class AuthServiceTests(unittest.TestCase):
    def test_operator_has_incident_permissions_but_not_settings(self):
        context = {"role": "operator"}
        self.assertTrue(has_permission(context, "update_incidents"))
        self.assertFalse(has_permission(context, "manage_settings"))

    def test_auditor_visible_sections_are_read_only(self):
        sections = [
            "Ситуационный центр",
            "Оперативный мониторинг",
            "Журнал инцидентов",
            "Аналитика и отчеты",
            "Подключение камер",
            "Настройки системы",
            "Доступ и аудит",
        ]
        visible = get_visible_sections(sections, {"role": "auditor"})
        self.assertIn("Ситуационный центр", visible)
        self.assertIn("Журнал инцидентов", visible)
        self.assertIn("Доступ и аудит", visible)
        self.assertNotIn("Подключение камер", visible)
        self.assertNotIn("Настройки системы", visible)


if __name__ == "__main__":
    unittest.main()

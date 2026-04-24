from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "vkr" / "screenshots"
BASE_URL = "http://localhost:8502"


def wait_for_app(page: Page) -> None:
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=120000)
    page.wait_for_timeout(7000)
    page.get_by_text("Enterprise Access Monitoring 24/7 Video Pipeline Neural Analytics").wait_for(timeout=120000)


def open_section(page: Page, section_name: str) -> None:
    page.get_by_text(section_name, exact=True).first.click()
    page.wait_for_timeout(3000)


def save_page(page: Page, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(OUT_DIR / filename), full_page=True)
    print(f"saved {filename}")


def capture_dashboard(page: Page) -> None:
    open_section(page, "Дашборд")
    save_page(page, "dashboard.png")


def capture_monitoring(page: Page) -> None:
    open_section(page, "Онлайн-мониторинг")
    page.wait_for_timeout(4000)
    save_page(page, "monitoring_focus.png")

    try:
        page.get_by_label("Режим отображения").click(timeout=3000)
        page.get_by_text("Сетка 2x2", exact=True).click(timeout=3000)
        page.wait_for_timeout(3000)
    except Exception:
        page.keyboard.press("Escape")
    save_page(page, "monitoring_grid_2x2.png")


def capture_live_window(browser) -> None:
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(
        f"{BASE_URL}/?view=live-window&source=production&source_id=1&source_kind=production&overlay=1",
        wait_until="domcontentloaded",
        timeout=120000,
    )
    page.wait_for_timeout(5000)
    save_page(page, "live_window.png")
    page.close()


def capture_journal(page: Page) -> None:
    open_section(page, "Журнал событий")
    save_page(page, "journal.png")


def capture_analytics(page: Page) -> None:
    open_section(page, "Аналитика")
    page.wait_for_timeout(4000)
    save_page(page, "analytics.png")


def capture_employees(page: Page) -> None:
    open_section(page, "Сотрудники")
    page.wait_for_timeout(4000)
    save_page(page, "employees.png")


def capture_sources(page: Page) -> None:
    open_section(page, "Источники видео")
    page.wait_for_timeout(3000)
    save_page(page, "sources.png")


def capture_settings(page: Page) -> None:
    open_section(page, "Настройки системы")
    page.wait_for_timeout(3000)
    save_page(page, "settings.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1400})
        wait_for_app(page)

        for func in [
            capture_dashboard,
            capture_monitoring,
            capture_journal,
            capture_analytics,
            capture_employees,
            capture_sources,
            capture_settings,
        ]:
            try:
                func(page)
            except Exception as exc:
                print(f"capture failed for {func.__name__}: {type(exc).__name__}: {exc}")
        try:
            capture_live_window(browser)
        except Exception as exc:
            print(f"capture failed for live window: {type(exc).__name__}: {exc}")

        browser.close()


if __name__ == "__main__":
    main()

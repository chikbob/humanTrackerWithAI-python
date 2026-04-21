"""WebRTC configuration helpers for browser live monitoring."""

from __future__ import annotations

import os


def _split_env_urls(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def build_rtc_configuration():
    """Build RTC configuration from environment variables for STUN/TURN deployment."""
    turn_urls = _split_env_urls("TURN_URLS")
    turn_username = os.getenv("TURN_USERNAME", "").strip()
    turn_password = os.getenv("TURN_PASSWORD", "").strip()
    stun_urls = _split_env_urls("STUN_URLS", "stun:stun.l.google.com:19302")

    ice_servers = []
    if stun_urls:
        ice_servers.append({"urls": stun_urls})
    if turn_urls:
        turn_config = {"urls": turn_urls}
        if turn_username:
            turn_config["username"] = turn_username
        if turn_password:
            turn_config["credential"] = turn_password
        ice_servers.append(turn_config)
    return {"iceServers": ice_servers} if ice_servers else None


def describe_rtc_environment() -> dict:
    """Return a human-readable diagnostic summary for browser live mode."""
    rtc_config = build_rtc_configuration()
    ice_servers = (rtc_config or {}).get("iceServers", [])
    has_turn = any(
        any(str(url).startswith("turn:") or str(url).startswith("turns:") for url in server.get("urls", []))
        for server in ice_servers
    )
    has_stun = any(
        any(str(url).startswith("stun:") for url in server.get("urls", []))
        for server in ice_servers
    )
    return {
        "rtc_config": rtc_config,
        "ice_server_count": len(ice_servers),
        "has_stun": has_stun,
        "has_turn": has_turn,
        "stun_urls": _split_env_urls("STUN_URLS", "stun:stun.l.google.com:19302"),
        "turn_urls": _split_env_urls("TURN_URLS"),
        "turn_username_configured": bool(os.getenv("TURN_USERNAME", "").strip()),
        "turn_password_configured": bool(os.getenv("TURN_PASSWORD", "").strip()),
    }

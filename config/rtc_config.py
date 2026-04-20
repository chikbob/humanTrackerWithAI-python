"""WebRTC configuration helpers for browser live monitoring."""

from __future__ import annotations

import os


def build_rtc_configuration():
    """Build RTC configuration from environment variables for STUN/TURN deployment."""
    turn_urls = [item.strip() for item in os.getenv("TURN_URLS", "").split(",") if item.strip()]
    turn_username = os.getenv("TURN_USERNAME", "").strip()
    turn_password = os.getenv("TURN_PASSWORD", "").strip()
    stun_urls = [item.strip() for item in os.getenv("STUN_URLS", "stun:stun.l.google.com:19302").split(",") if item.strip()]

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

"""Shared mapping state enums and helpers."""

from __future__ import annotations

from enum import IntEnum


class NavStatus(IntEnum):
    IDLE = 0
    NAVIGATING = 1
    REACHED = 2
    FAILED = 3
    STUCK = 4


class ExplorationStatus(IntEnum):
    WAITING_FOR_MAP = 0
    EXPLORING = 1
    COMPLETE = 2
    TIMEOUT = 3
    FAILED = 4

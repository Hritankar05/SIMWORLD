"""Portable column types that work with both PostgreSQL and SQLite.

PostgreSQL uses native UUID and JSONB types.
SQLite uses CHAR(36) for UUIDs and TEXT/JSON for JSON columns.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import String, Text, TypeDecorator


class PortableUUID(TypeDecorator):
    """Platform-agnostic UUID type.

    Uses CHAR(36) storage which works everywhere.
    Converts between Python uuid.UUID and string representations.
    """

    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return str(value)
        return str(value)

    def process_result_value(self, value: Any, dialect: Any) -> uuid.UUID | None:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class PortableJSON(TypeDecorator):
    """Platform-agnostic JSON type.

    Uses TEXT storage with JSON serialization/deserialization.
    Works with both PostgreSQL (native JSON) and SQLite (TEXT).
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        return json.loads(value)

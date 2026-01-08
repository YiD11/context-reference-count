"""Serialization utilities for tool arguments."""

import json
from datetime import datetime
from typing import Any, override


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    
    @override
    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def serialize_args(args: dict[str, Any]) -> str:
    """Serialize tool arguments to a canonical string representation."""
    return json.dumps(args, sort_keys=True, cls=DateTimeEncoder)


def deserialize_args(args_str: str) -> dict[str, Any]:
    """Deserialize tool arguments from string."""
    return json.loads(args_str)

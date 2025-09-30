from dataclasses import dataclass, asdict
from typing import Literal


@dataclass
class IffyIndex:
    status: Literal["ok", "check", "iffy", "not_assessed"]
    explanation: str

    def icon(self) -> str:
        return {
            "ok": "✅",
            "check": "⚠️",
            "iffy": "❌",
            "not_assessed": "⚪"
        }[self.status]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["icon"] = self.icon()
        return d

    def __str__(self) -> str:
        return f"{self.icon()} {self.status} — {self.explanation}"

    def __repr__(self) -> str:
        return self.__str__()
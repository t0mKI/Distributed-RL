from enum import Enum
from typing import Any


class StringEnum(Enum):

    def __new__(cls, *args) -> Enum:
        member = object.__new__(cls)
        member._value_ = args[0]
        return member

    def __str__(self) -> str:
        # converts enum into string representation (equivalent to using accessing member "value" in Python 3.X)
        return self._value_

    def to_value(self) -> Any:
        return self._value_

    @classmethod
    def get_index(cls, type):
        return list(cls).index(type)

class IntEnum(Enum):

    def __new__(cls, *args) -> Enum:
        member = object.__new__(cls)
        member._value_ = args[0]
        return member

    def to_value(self) -> int:
        # converts enum into integer representation (equivalent to using IntEnum)
        return self._value_


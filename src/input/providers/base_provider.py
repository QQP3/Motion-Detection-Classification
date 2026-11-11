
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseProvider(ABC):
    """Common interface for all video input providers."""

    def __init__(self, name: str):
        self.name = name
        self.is_open = False

    @abstractmethod
    def open(self) -> bool:
        """Open the input source."""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release input resources."""
        pass

    def is_opened(self) -> bool:
        """Return whether the provider is open."""
        return self.is_open

    def __enter__(self):
        if not self.open():
            raise RuntimeError(
                f"Could not open input provider: {self.name}"
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False

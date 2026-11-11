
from typing import Dict, Optional, Tuple

import numpy as np

from src.input.providers.base_provider import BaseProvider


class InputManager:
    """Manage multiple video input providers."""

    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}

    def add_provider(self, provider: BaseProvider) -> None:
        """Register a new input provider."""

        if provider.name in self.providers:
            raise ValueError(
                f"Provider '{provider.name}' already exists."
            )

        self.providers[provider.name] = provider

    def get_provider(self, name: str) -> BaseProvider:
        """Retrieve a registered provider."""

        if name not in self.providers:
            raise KeyError(f"Provider '{name}' not found.")

        return self.providers[name]

    def open_provider(self, name: str) -> bool:
        """Open one registered provider."""

        provider = self.get_provider(name)
        return provider.open()

    def open_all(self) -> Dict[str, bool]:
        """Attempt to open every registered provider."""

        results = {}

        for name, provider in self.providers.items():
            try:
                results[name] = provider.open()
            except Exception:
                results[name] = False

        return results

    def read(
        self, name: str
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from one provider."""

        provider = self.get_provider(name)
        return provider.read()

    def read_all(
        self,
    ) -> Dict[str, Tuple[bool, Optional[np.ndarray]]]:
        """Read one frame from every opened provider."""

        results = {}

        for name, provider in self.providers.items():
            if not provider.is_opened():
                continue

            try:
                results[name] = provider.read()
            except Exception:
                results[name] = (False, None)

        return results

    def release_provider(self, name: str) -> None:
        """Release one provider."""

        provider = self.get_provider(name)
        provider.release()

    def release_all(self) -> None:
        """Release all registered providers."""

        for provider in self.providers.values():
            try:
                provider.release()
            except Exception:
                pass

    def __enter__(self):
        self.open_all()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release_all()
        return False

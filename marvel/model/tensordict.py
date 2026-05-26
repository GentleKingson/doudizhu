from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping

import torch


class TensorDict:
    """Lightweight TensorDict shim for torch-only runtimes."""

    def __init__(
        self,
        data: Mapping[str, Any],
        *,
        batch_size: Iterable[int] | torch.Size,
        device: torch.device | str | None = None,
    ) -> None:
        resolved_device = torch.device(device) if device is not None else None
        converted: Dict[str, torch.Tensor] = {}
        for key, value in data.items():
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if resolved_device is not None and tensor.device != resolved_device:
                tensor = tensor.to(resolved_device)
            converted[key] = tensor
        object.__setattr__(self, "_data", converted)
        object.__setattr__(self, "_batch_size", torch.Size(batch_size))

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        if not self._data:
            return torch.device("cpu")
        return next(iter(self._data.values())).device

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def items(self) -> Iterable[tuple[str, torch.Tensor]]:
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> "TensorDict":
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        self._data[key] = tensor
        return self

    def set_(self, key: str, value: Any) -> "TensorDict":
        return self.set(key, value)

    def select(self, *keys: str) -> "TensorDict":
        selected = {key: self._data[key] for key in keys}
        return TensorDict(selected, batch_size = self._batch_size, device = self.device)

    def to(self, *args, **kwargs) -> "TensorDict":
        converted = {key: value.to(*args, **kwargs) for key, value in self._data.items()}
        device = None
        if converted:
            device = next(iter(converted.values())).device
        return TensorDict(converted, batch_size = self._batch_size, device = device)

    def clone(self) -> "TensorDict":
        cloned = {key: value.clone() for key, value in self._data.items()}
        return TensorDict(cloned, batch_size = self._batch_size, device = self.device)

    def _slice_batch(self, index: Any) -> "TensorDict":
        sliced = {key: value[index] for key, value in self._data.items()}
        if sliced:
            sample = next(iter(sliced.values()))
            if sample.dim() == 0:
                batch_size = torch.Size([])
            else:
                batch_size = torch.Size([sample.shape[0]])
        else:
            batch_size = self._batch_size
        return TensorDict(sliced, batch_size = batch_size, device = self.device)

    def _assign_batch(self, index: Any, other: "TensorDict") -> None:
        if not isinstance(other, TensorDict):
            raise TypeError("Assigned value must be a TensorDict.")
        for key, value in other.items():
            if key not in self._data:
                raise KeyError(f"Missing key '{key}' in target TensorDict.")
            self._data[key][index] = value

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            return self._data[item]
        return self._slice_batch(item)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            self.set(key, value)
            return
        if isinstance(value, TensorDict):
            self._assign_batch(key, value)
            return
        if isinstance(value, MutableMapping):
            self._assign_batch(key, TensorDict(value, batch_size = self._batch_size, device = self.device))
            return
        raise TypeError("Assigned value must be a TensorDict or mapping.")

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute '{name}'.")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        self._data[name] = tensor

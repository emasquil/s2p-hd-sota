from __future__ import annotations

import abc
import logging
import math
import multiprocessing
import multiprocessing.context
import multiprocessing.sharedctypes
import multiprocessing.synchronize
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

logger = logging.getLogger(__name__)

MEM_SLICE_PER_TOKEN = 100


class UnavailableMemoryException(Exception):
    """The device does not have enough total memory to handle a request."""

    pass


class GPUMemoryManager(abc.ABC):
    @abc.abstractmethod
    @contextmanager
    def request(self, megabytes: float) -> Generator[None, None, None]:
        ...

    @staticmethod
    def make_bounded(
        max_memory_in_megabytes: float,
        mp_context: multiprocessing.context.BaseContext,
    ) -> GPUMemoryManager:
        num_tokens = math.floor(max_memory_in_megabytes // MEM_SLICE_PER_TOKEN)
        counter = mp_context.Value("i", num_tokens)
        return _BoundedGPUMemoryManager(
            num_tokens=num_tokens,
            counter=counter,
        )

    @staticmethod
    def make_unbounded() -> _UnboundedGPUMemoryManager:
        return _UnboundedGPUMemoryManager()


@dataclass
class _BoundedGPUMemoryManager(GPUMemoryManager):
    num_tokens: int
    """ 1 token = 100MB """
    counter: Any

    @contextmanager
    def request(self, megabytes: float) -> Generator[None, None, None]:
        # add some overhead just in case
        megabytes += megabytes * 0.05

        tokens = math.ceil(megabytes / MEM_SLICE_PER_TOKEN)

        if tokens > self.num_tokens:
            raise UnavailableMemoryException(
                f"{tokens} tokens requested, only {self.num_tokens} total available"
                f" (1 token = {MEM_SLICE_PER_TOKEN}MB)"
            )

        # this is a poor's man semaphore, because python's semaphore does not support acquiring many tokens at once
        # and doing something smarter with Condition harder to get right.
        while True:
            with self.counter.get_lock():
                if self.counter.value >= tokens:
                    self.counter.value -= tokens
                    break
            logging.info(f"waiting for {tokens} tokens (total {self.num_tokens})")
            time.sleep(0.1)

        try:
            logging.info(f"we got the tokens, lets work")
            yield
            logging.info(f"we worked, lets release the tokens")
        finally:
            logging.info(f"releasing {tokens} tokens")
            with self.counter.get_lock():
                self.counter.value += tokens


@dataclass
class _UnboundedGPUMemoryManager(GPUMemoryManager):
    @contextmanager
    def request(self, megabytes: float) -> Generator[None, None, None]:
        yield

import asyncio
from typing import List

class AsyncBatchQueue:
    def __init__(self, max_batch_size: int = 8, timeout_ms: int = 50):
        self.max_batch_size = max_batch_size
        self.timeout = timeout_ms / 1000.0

    async def process_batch(self, items: List):
        return [f"Caption {i}" for i in range(len(items))]

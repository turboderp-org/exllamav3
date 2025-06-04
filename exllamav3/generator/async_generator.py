from __future__ import annotations
from .generator import Generator
from .job import Job
import asyncio

class AsyncGenerator:
    """
    Async wrapper for dynamic generator. See definition of Generator.
    """
    def __init__(self, *args, **kwargs):
        self.generator = Generator(*args, **kwargs)
        self.jobs = {}
        self.condition = asyncio.Condition()
        self.iteration_task = asyncio.create_task(self._run_iteration())

    async def _run_iteration(self):
        try:
            while True:
                async with self.condition:
                    # Unlock if there's no jobs or if the parent task is cancelled
                    await self.condition.wait_for(lambda: len(self.jobs) > 0 or self.iteration_task.cancelled())

                results = self.generator.iterate()
                for result in results:
                    job = result["job"]
                    async_job = self.jobs[job]
                    await async_job.put_result(result)
                    if result["eos"]:
                        del self.jobs[job]
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            # Silently return on cancel
            return
        except Exception as e:
            # If the generator throws an exception it won't pertain to any one ongoing job, so push it to all of them
            for async_job in self.jobs.values():
                await async_job.put_result(e)

    def enqueue(self, job: AsyncJob):
        assert job.job not in self.jobs
        self.jobs[job.job] = job
        self.generator.enqueue(job.job)
        asyncio.create_task(self._notify_condition())

    async def _notify_condition(self):
        async with self.condition:
            self.condition.notify_all()

    async def close(self):
        self.iteration_task.cancel()

        # Force a re-check of the condition to unlock the loop
        await self._notify_condition()
        try:
            await self.iteration_task
        except asyncio.CancelledError:
            pass

    async def cancel(self, job: AsyncJob):
        assert job.job in self.jobs
        self.generator.cancel(job.job)
        del self.jobs[job.job]


class AsyncJob:
    """
    Async wrapper for dynamic generator job. See definition of Job.
    """
    def __init__(self, generator: AsyncGenerator, *args: object, **kwargs: object):
        self.generator = generator
        self.job = Job(*args, **kwargs)
        self.queue = asyncio.Queue()
        self.generator.enqueue(self)
        self.cancelled = False

    async def put_result(self, result):
        await self.queue.put(result)

    async def __aiter__(self):
        while True:
            # Get out if the job is cancelled
            if self.cancelled:
                break

            result = await self.queue.get()
            if isinstance(result, Exception):
                raise result
            yield result
            if result["eos"]:
                break

    async def cancel(self):
        await self.generator.cancel(self)
        self.cancelled = True

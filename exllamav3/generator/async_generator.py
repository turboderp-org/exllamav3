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
                # Sleep while there is no async work registered. The condition releases its lock while waiting and
                # is notified by enqueue() or close(), so this background task does not spin between requests.
                async with self.condition:
                    # Wake when the first job arrives or when close() has cancelled the iteration task.
                    await self.condition.wait_for(lambda: len(self.jobs) > 0 or self.iteration_task.cancelled())

                # Drive exactly one synchronous generator step and fan out any returned events to the owning
                # AsyncJob queues. Missing jobs can happen if a job was cancelled after iterate() started.
                results = self.generator.iterate()
                for result in results:
                    job = result["job"]
                    async_job = self.jobs.get(job)
                    if not async_job:
                        continue
                    await async_job.put_result(result)
                    if result["eos"]:
                        del self.jobs[job]

                # Yield back to the event loop so result consumers and cancellation requests can run between
                # generator iterations, even if the synchronous generator immediately has more work available.
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            # Silently return on cancel
            return

        except Exception as e:
            # If the generator throws an exception it won't pertain to any one ongoing job, so push it to all of them
            for async_job in self.jobs.values():
                await async_job.put_result(e)

    def enqueue(self, job: AsyncJob):
        # Track the AsyncJob before enqueueing the underlying Job so any immediate generator result has a queue to
        # land in. The sync generator still owns scheduling and serial assignment.
        assert job.job not in self.jobs
        self.jobs[job.job] = job
        self.generator.enqueue(job.job)

        # Condition.notify_all() must run while holding the condition lock, so schedule a tiny coroutine instead of
        # trying to notify directly from this synchronous method.
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
        # Remove the underlying Job from the synchronous generator first so no new tokens are produced, then drop
        # the async mapping so any late results from an in-flight iteration are ignored.
        self.generator.cancel(job.job)
        if job.job not in self.jobs:
            return
        del self.jobs[job.job]


class AsyncJob:
    """
    Async wrapper for dynamic generator job. See definition of Job.
    """
    def __init__(self, generator: AsyncGenerator, *args: object, **kwargs: object):
        self.generator = generator
        self.job = Job(*args, **kwargs)
        self.queue = asyncio.Queue(maxsize=16)
        self.generator.enqueue(self)
        self.cancelled = False

    async def put_result(self, result):
        await self.queue.put(result)

    async def __aiter__(self):
        while True:
            # cancel() sets this flag after removing the job from the generator; the iterator exits cleanly instead
            # of waiting forever for an EOS event that will no longer be produced.
            if self.cancelled:
                break

            # Results are pushed by AsyncGenerator._run_iteration(). Exceptions are broadcast to all queues if the
            # shared generator loop fails, so consuming code sees the error at the await point.
            result = await self.queue.get()
            if isinstance(result, Exception):
                raise result
            yield result
            if result["eos"]:
                break

    async def cancel(self):
        # Delegate cancellation to the wrapper so it can update both the sync generator queue and the async job map,
        # then mark this iterator as closed for any current or future consumers.
        await self.generator.cancel(self)
        self.cancelled = True

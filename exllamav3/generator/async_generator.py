from __future__ import annotations
from .generator import Generator
from .job import Job
import asyncio

# Sentinel pushed to an AsyncJob's queue on cancellation so a consumer parked in queue.get() wakes up and
# exits, rather than waiting forever for results that will no longer be produced
_CANCELLED_SENTINEL = object()

class AsyncGenerator:
    """
    Async wrapper for dynamic generator. See definition of Generator.
    """
    def __init__(self, *args, **kwargs):
        self.generator = Generator(*args, **kwargs)
        self.jobs = {}
        self.error = None
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
                # Delivery must never block: this single task serves every job, so waiting on one stalled
                # consumer (e.g. a disconnected client that stopped draining its queue) would wedge the whole
                # generator (issue #227).
                results = self.generator.iterate()
                for result in results:
                    job = result["job"]
                    async_job = self.jobs.get(job)
                    if not async_job:
                        continue
                    async_job.put_result(result)
                    if result["eos"]:
                        del self.jobs[job]

                # Yield back to the event loop so result consumers and cancellation requests can run between
                # generator iterations, even if the synchronous generator immediately has more work available.
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            # Silently return on cancel
            return

        except Exception as e:
            # If the generator throws an exception it won't pertain to any one ongoing job, so push it to all of
            # them. The iteration task ends here and the generator is unusable, so record the error and deliver
            # it to any job enqueued later as well, instead of letting those jobs wait forever on a dead loop.
            self.error = e
            for async_job in self.jobs.values():
                async_job.put_result(e)
            self.jobs.clear()

    def enqueue(self, job: AsyncJob):
        # The iteration task died on a generator exception; surface it to the new job's consumer immediately
        if self.error is not None:
            job.put_result(self.error)
            return

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

        # Wake any consumers still parked on job queues; no more results will be produced
        for async_job in self.jobs.values():
            async_job.put_result(_CANCELLED_SENTINEL)
        self.jobs.clear()

    async def cancel(self, job: AsyncJob):
        # Remove the underlying Job from the synchronous generator first so no new tokens are produced, then drop
        # the async mapping so any late results from an in-flight iteration are ignored. Finally wake the job's
        # consumer in case it is parked in queue.get(), waiting for a result that will never arrive.
        # After a generator exception the sync generator's state is suspect and every job was already delivered
        # the error; don't poke it from consumers' cleanup handlers, or their except/finally blocks can be hit
        # with a second exception.
        if self.error is None:
            self.generator.cancel(job.job)
        if job.job not in self.jobs:
            return
        del self.jobs[job.job]
        job.put_result(_CANCELLED_SENTINEL)


class AsyncJob:
    """
    Async wrapper for dynamic generator job. See definition of Job.
    """
    def __init__(self, generator: AsyncGenerator, *args: object, **kwargs: object):
        self.generator = generator
        self.job = Job(*args, **kwargs)
        # The queue is unbounded and written with put_nowait: the shared iteration task must never block on a
        # consumer that stopped draining (see _run_iteration). Growth is paced by the sync generator's own token
        # rate and ends when the job completes or is cancelled, which is the frontend's responsibility on client
        # disconnect.
        self.queue = asyncio.Queue()
        self.generator.enqueue(self)
        self.cancelled = False

    def put_result(self, result):
        self.queue.put_nowait(result)

    async def __aiter__(self):
        while True:
            # cancel() sets this flag after removing the job from the generator; the iterator exits cleanly instead
            # of waiting forever for an EOS event that will no longer be produced.
            if self.cancelled:
                break

            # Results are pushed by AsyncGenerator._run_iteration(). Exceptions are broadcast to all queues if the
            # shared generator loop fails, so consuming code sees the error at the await point. The sentinel wakes
            # this iterator when the job is cancelled from another task while parked here.
            result = await self.queue.get()
            if result is _CANCELLED_SENTINEL:
                break
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

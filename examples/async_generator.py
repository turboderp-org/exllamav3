import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Model, Config, Cache, Tokenizer, AsyncGenerator, AsyncJob, Sampler
import asyncio

"""
The async generator is a wrapper class that allows you to treat generator jobs as asynchronous iterators, while
still letting concurrent jobs benefit from batching. Here is a simple example using asyncio.gather to launch a
batch of async tasks at once.
"""

async def main():

    # Load model etc.
    config = Config.from_directory("/mnt/str/eval_models/llama3.1-8b-instruct/exl3/4.0bpw/")
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 32768)
    model.load()
    tokenizer = Tokenizer.from_config(config)

    # Initialize the async generator with default settings
    generator = AsyncGenerator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
    )

    # Define a couple of prompts
    prompts = [
        "Once upon a time, there was",
        "asyncio in Python is a great feature because",
        "asyncio in Python is a pain to work with because",
    ]

    # Async task running async job in the async generator
    async def run_job(prompt: str, marker: str):

        # Create an asynchronous job. The job presents as an iterator which is transparently batched with other
        # concurrent jobs for the same generator.
        job = AsyncJob(
            generator,
            input_ids = tokenizer.encode(prompt, add_bos = False),
            max_new_tokens = 200
        )

        # Iterate over the job. Each returned result is a dictionary containing an update on the status of the
        # job and/or part of the completion (see the definition of Job.iterate() for details). The iterator ends
        # when the job is complete (i.e. EOS or max_new_tokens is reached)
        full_completion = prompt
        async for result in job:
            # We'll only collect text here, but the result could contain other updates
            full_completion += result.get("text", "")

            # Output marker to console to confirm that tasks running asynchronously, and that job 0 stops running
            # after 300 characters (note, not tokens)
            print(marker, end = "", flush = True)

            # Cancel the second job after 300 characters to make the control flow less trivial. We have to explicitly
            # cancel the job, otherwise the generator will continue to run the job in the background, waiting for some
            # task to finish iterating through the results
            if marker == "1" and len(full_completion) > 300:
                full_completion += " [job canceled]"
                await job.cancel()
                break
        else:
            full_completion += " [max_new_tokens reached]"

        return full_completion

    # Run a batch of async jobs
    tasks = [run_job(prompt, str(i)) for i, prompt in enumerate(prompts)]
    outputs = await asyncio.gather(*tasks)

    # Print the results
    print()
    print()
    for i, output in enumerate(outputs):
        print(f"Output {i}")
        print("-----------")
        print(output)
        print()

    await generator.close()

if __name__ == "__main__":
    asyncio.run(main())




import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3 import Config, Model, Cache, Tokenizer

# Define model by loading the config from the model directory
config = Config.from_directory("/mnt/str/models/llama3.1-8b-instruct/exl3/4.0bpw/")

# Create and load tokenizer
tokenizer = Tokenizer.from_config(config)

# Create the model instance. Model isn't loaded until one of the load functions is called
model = Model.from_config(config)

# Load model without cache (then unload)
print("Loading")
model.load()
model.unload()

# Load model with progress bar
print("Loading with progress bar")
model.load(progressbar = True)
model.unload()

# Create a cache attached to the model. When the model is loaded, the cache tensors are also created
cache = Cache(model, max_num_tokens = 2048)

# Load again, this time with a cache
print("Loading with cache")
model.load(progressbar = True)
model.unload()

# Unloading the model also destroys the tensors of any attached cache(s), freeing up all the VRAM used by both.
# The cache is still attached to the model at this point, so loading the same model again would create the
# cache tensors anew. If desired, we can explicitly detach the cache to prevent this:
cache.detach_from_model(model)

# Load with callback function
def progress_callback(module: int, modules: int):
    print(f"Callback: Loaded {module} of {modules} modules")

print("Loading with callback")
model.load(callback = progress_callback)
model.unload()

# Load model using generator function. In this mode, the loader presents as an iterator yielding the current
# progress on each iteration
print("Loading with generator function")
f = model.load_gen()
for module, modules in f:
    print(f"Generator: Loaded {module} of {modules} modules")
model.unload()

# Load model using generator function, callback and progress bar
print("Loading with generator function, callback and progress bar")
f = model.load_gen(progressbar = True, callback = progress_callback)
for module, modules in f:
    print(f"Generator: Loaded {module} of {modules} modules")
model.unload()
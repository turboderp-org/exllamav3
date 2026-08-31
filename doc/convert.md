## EXL3 conversion script

### Arguments

#### Basic

- **-i / --in_dir *directory***: The source model to convert, in unquantized HF format. The directory should contain at least a `config.json` file, a `tokenizer.json` file and one or more `.safetensors` files containing weights. 
  
- **-o / --out_dir *directory***: The destination directory for the converted **EXL3** model. Will be created if it doesn't exist, or overwritten if it
does.

- **-w / --work_dir *directory***: Working directory for temporary files. It should have enough free space to store an entire copy of the output model. This is also where checkpoints are stored and the only required argument if **-r / --resume** is specified.

- **-ss / --shard_size *float***: Output shard size, in megabytes. Default is 8192. Set this to 0 to disable sharding. Note that writing very large `.safetensors` files can require a lot of system RAM.

- **-b / --bits *float***: Target average number of bits per weight.
  
- **-hb / --head_bits *int***: Number of bits per weight for the lm_head (output) layer of the model. Must be an integer from 1 to 8, default is 6.

- **-mb / --mtp_bits *int***: Number of bits per weight for the MTP (multi-token prediction) layers, for models that have them. Must be an integer from 1 to 8, or 16 to store the MTP layers unquantized. Default is 4. MTP layers are quantized without calibration.

- **-vb / --vision_bits *int***: Number of bits per weight for the vision component, for models that have one. Must be an integer from 1 to 8, or 16 to copy the vision model unquantized. The default depends on the architecture: vision towers validated for low-bitrate quantization default to 6, everything else to 16. Vision layers are quantized without calibration.

- **-ngb / --ngram_bits *int***: Bits per weight for hashed n-gram embedding tables (PLE models, e.g. Qwen3.8-Flash-Next). Must be an integer from 1 to 8, default is `--bits` rounded to the nearest integer. The table is quantized (without calibration) before the layer-by-layer conversion and written as a standalone `ngram_embedding.safetensors` in the output directory; calibration forwards then run against the quantized table. Table quantization is resumable together with the rest of the job.

- **-ngf / --ngram_file *file***: Pre-quantized n-gram table (from `util/convert_ngram.py`) to copy into the output model instead of quantizing the table as part of the job. Overrides `--ngram_bits`.

- **-hq / --hq**: Increase the bitrate of select layers, such as attention and shared-expert layers. Final model bitrate may be somewhat higher than requested by `--bits`, but for MoE models this is typically a very small increase in size (0.05 - 0.10 bpw) for a disproportionately large increase in model fidelity. 

- **-rcp / --recipe *file***: Per-tensor bitrate recipe (YAML, e.g. from `sc_optimize.py`), used in place of the budgeted allocation from `--bits`/`--head_bits`. More about optimized recipes [here](optimize.md).

- **-cd / --cal_data *file***: Calibration data file (safetensors with packed token rows, e.g. from `sc_trace.py`) used instead of the bundled corpus mix. 

#### Advanced (generally disregard these options)

- **--out_scales *str***: Force enable or disable output channel scales. Options are "always" (default), "never" and "auto". Mostly for debug purposes. 

- **-cb / --codebook *str***: Trellis codebook: "mul1" (default), "mcg" or "3inst". The mul1 codebook is required by some optimized inference paths (int8 GEMV, CPU expert offload); there is no reason to pick another codebook except for testing.

#### Checkpoints

- **-cpi / --checkpoint_interval *int***: Minimum interval (in seconds) between checkpoints. Default is 120.

- **-r / --resume**: Resume an interrupted job pointed to by **-w / --work_dir** from the latest checkpoint. If resuming a job, all other arguments such as input and output directories, bitrate etc. are restored from the old job, though some can be overridden. Note that resuming is now explicit, reversing the behavior from ExLlamaV2.

#### Performance

- **-d / --devices *list***: Comma-separated list of GPU device IDs to use during quantization. By default only the first visible device (device 0) is used. Adding more devices can speed up quantization if there is sufficient PCIe bandwidth between them. This does not affect memory usage on the first GPU, and very little memory is used on the others, since only the most compute intensive operation (trellis encoding) is distributed.

- **-dr / --device_ratios *list***: Ratio as comma-separated list. Determines how the encoding workload is distributed when using multiple devices. This is useful if using GPUs with dissimilar compute performance, to prevent slower GPUs from becoming bottlenecks. Ratios are relative, i.e. `1,1,3` is the same ratio as `3,3,9`. Recommendation is to omit this argument; by default, ratios are autotuned to maximize usage across GPUs.

- **-pm / --parallel_mode**: Deprecated (no-op). Parallel mode is now the default: multi-GPU quantization distributes one linear layer to each GPU at a time whenever a layer has at least as many tensors as there are devices. Layers with fewer tensors than devices (e.g. the lm_head, alone in its layer) fall back to splitting the trellis encoding workload across devices, with small tensors capped to as many devices as they can feed (~1M weights per device).

#### Debug stuff (ignore these)

- **-lcpi / --last_checkpoint_index *int***: If specified, don't save checkpoints after this module index.

- **-cr / --cal_rows *int***: Number of rows of calibration data. Default is 250.

- **-cc / --cal_cols *int***: Number of columns of calibration data. Default is 2048.

- **-v / --verbose**: Extra debug output while quantizing.

- **--override_anyway**: Allow resuming even when overriding settings that will break the existing job. 

- **-img / --image_dump**: Save all tensors as images in the working directory. May require a large amount of system memory and disk space, can be slow.  

- **--max_module *int***: End quantization after this many modules, including embedding and norm layers.

### Examples

#### Converting

```sh
python convert.py -i /mnt/models/llama3.1-70b-instruct \
                  -o /mnt/models/llama3.1-70b-instruct-exl3-3.75bpw \
                  -w /mnt/temp/exl3 \
                  -b 3.75
```

#### Resuming

Resume the job started above if it was interrupted:

```sh
python convert.py -w /mnt/temp/exl3 -r
```

#### Multi-GPU quant

Convert a model on the first three devices:

```sh
python convert.py -i /mnt/models/llama3.1-70b-instruct \
                  -o /mnt/models/llama3.1-70b-instruct-exl3-3.75bpw \
                  -w /mnt/temp/exl3 \
                  -b 3.75 \
                  -d 0,1,2
```

Convert a model on the first three devices, using CUDA:2 as the primary device. Also keep attention etc. in higher precision with `-hq`. Final model size will be slightly larger than the 4.00 bpw requested in this example, but since this is an MoE model, the increase will be on the order of 0.05 - 0.1 bpw:

```sh
python convert.py -i /mnt/models/qwen3.5-35b-a3b \
                  -o /mnt/models/qwen3.5-35b-a3b-4.00bpw-plus \
                  -w /mnt/temp/exl3 \
                  -b 4.00 \
                  -hq \
                  -d 2,0,1
```

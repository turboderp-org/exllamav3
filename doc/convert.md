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

#### Advanced (generally disregard these options)

- **--out_scales | --no_out_scales**: Force enable or disable output channel scales. By default, whether to apply this setting is autodetected during quantization, so don't set either of these flags unless there's a good reason to. Mostly they're for debug purposes. 

#### Checkpoints

- **-cpi / --checkpoint_interval *int***: Interval (in seconds) between checkpoints.

- **-r / --resume**: Resume an interrupted job pointed to by **-w / --work_dir** from the latest checkpoint. If resuming a job, all other arguments such as input and output directories, bitrate etc. are restored from the old job, though some can be overridden. Note that resuming is now explicit, reversing the behavior from ExLlamaV2.

#### Performance

- **-d / --devices *list***: Comma-separated list of GPU device IDs to use during quantization. By default only the first visible device (device 0) is used. Adding more devices can speed up quantization if there is sufficient PCIe bandwidth between them. This does not affect memory usage on the first GPU, and very little memory is used on the others, since only the most compute intensive operation (trellis encoding) is distributed.

- **-dr / --device_ratios *list***: Ratio as comma-separated list. Determines how the encoding workload is distributed when using multiple devices. This is useful if using GPUs with dissimilar compute performance, to prevent slower GPUs from becoming bottlenecks. Ratios are relative, i.e. `1,1,3` is the same ratio as `3,3,9`.

#### Debug stuff (ignore these)

- **-lcpi / --last_checkpoint_index *int***: If specified, don't save checkpoints after this module index.

- **-cr / --cal_rows *int***: Number of rows of calibration data.

- **-cc / --cal_cols *int***: Number of columns of calibration data.

- **-v / --verbose**: Extra debug output while quantizing.

- **--override_anyway**: Allow resuming even when overriding settings that will break the existing job. 

- **-img / image_dump**: Save all tensors as images in the working directory. May require a large amount of system memory and disk space, can be slow.  

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

Convert a model on the first three devices, using CUDA:2 as the primary device and distributing 3/12, 4/12 and 5/12 of the workload to CUDA:2, CUDA:0 and CUDA:1, respectively:

```sh
python convert.py -i /mnt/models/llama3.1-70b-instruct \
                  -o /mnt/models/llama3.1-70b-instruct-exl3-3.75bpw \
                  -w /mnt/temp/exl3 \
                  -b 3.75 \
                  -d 2,0,1 \
                  -dr 3,4,5
```
 
For dialing in the optimal ratio, monitor GPU usage while quantizing. Usage should periodically jump to 100% for at least one device, and ideally you want the other devices close to that as well, while they are active. Increasing the relative split for a device should increase the relative usage as well.

### Expected duration

Some rough estimates of the expected overall time to convert various sizes of models, as of v0.0.1. Quantization kernel is tuned for Ada GPUs at the moment, and especially Ampere performance is likely to improve with more optimization.

| Model size | bpw | 3090   | 4090   | 5090   | 5090 + 2x4090 |
|------------|-----|--------|--------|--------|---------------|
| 1B         | 2.0 | 11m    | 4m     | 3m     | 2m            |
| 1B         | 4.0 | 6m     | 3m     | 2m     | 1m            |
| 1B         | 8.0 | 5m     | 3m     | 2m     | 1m            |
| 8B         | 2.0 | 1h 08m | 21m    | 16m    | 10m           |
| 8B         | 4.0 | 34m    | 14m    | 12m    | 8m            |
| 8B         | 8.0 | 31m    | 14m    | 11m    | 8m            |
| 70B        | 2.0 | 9h 04m | 3h 7m  | 2h 19m | 1h 24m        |
| 70B        | 4.0 | 4h 32m | 2h 0m  | 1h 50m | 1h 5m         |
| 70B        | 8.0 | 4h 20m | 1h 58m | 1h 41  | 1h 3m         |


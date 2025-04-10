
# <img src="doc/cat.png" width="40"> ExlLlamaV3

This is an **early preview release** of ExLlamaV3. Please note: ↙

- The framework <u>is not yet fully optimized</u>. Performance is lacking, especially on Ampere, and there may be a significant CPU bottleneck on slower processors until the extension functions are fully built out.
- AMD GPUs (ROCm) are not yet supported.
- [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) is currently required. I hope to switch over to [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/main) in time, but there are some obstacles to overcome first. 
- A number of important features are yet to be added, such as cache quantization, tensor parallelism and multimodal support.
- There are no release builds yet.
- Integration into [TabbyAPI](https://github.com/theroyallab/tabbyAPI/) is planned when all the core functionality is in place.

## Why?

As the name implies, the original intention for ExLlama was to run inference on quantized Llama models. ExLlamaV2 was able to support a number of other architectures by treating every new model as (more or less) a Llama variant with optional features. However, as new models are increasingly moving away from the basic transformer template, this approach is no longer sustainable.  

Additionally, ExLlamaV2 is largely designed to run in a single process and CUDA doesn't like this very much when spreading a workload across multiple GPUs. It's a fundamental design feature in the CUDA runtime, and it has become a major obstacle to tensor-parallel inference, demand for which seems to keep increasing. This shortcoming is not easily addressed without a rewrite. Moreover, the **EXL2** format doesn't lend itself well to parallel inference in the first place due its input channel permutation.

 Aside from lifting a few of the most successful features from V2 (such as the generator), ExLlamaV3 is largely rewritten from scratch to provide a cleaner, more modular framework for supporting newer architectures. It also introduces a new SOTA quantization format based on [**QTIP**](https://github.com/Cornell-RelaxML/qtip) (see below).

## What's missing?

There's much that still needs to be added and/or ported over from ExLlamaV2. I've decided to release ExLlamaV3 in its current state to invite testing, feedback and contributions, but please be aware that it's not yet a viable replacement for ExLlamaV2. Currently on the to-do list:

- Support for more architectures (Mixtral, Cohere and Deepseek are in the works)
- Samplers (most notably repetition penalties and min-P are missing)
- Constrained sampling (JSON filters etc.)
- Multimodal support
- Cache quantization
- LoRA support
- ROCm support
- Tensor-parallel inference
- Lots of optimization

As for what is implemented, expect that some things may be a little broken at first. Please be patient and/or contribute. 👉👈 

## How to?

### Installation

Detailed installation instructions are coming soon, along with prebuilt wheels. For the time being, you can install the library with:

```sh
# Full installation
pip install -r requirements.txt
pip install .

# JIT mode
EXLLAMA_NOCOMPILE=1 pip install . 
```

Note that the included scripts can run in JIT mode from the repo directory without installing the library.

### Conversion

To convert a model to EXL3 format, use:

```sh
# Convert model
python convert.py -i <input_dir> -o <output_dir> -w <working_dir> -b <bitrate>

# Resume an interrupted quant job
convert.py -w <working_dir> -r

# More options
convert.py -h
```

The working directory is temporary storage for state checkpoints and for storing quantized tensors until the converted model can be compiled. It should have enough free space to store an entire copy of the output model. Note that while EXL2 conversion by default resumes an interrupted job when pointed to an existing folder, EXL3 needs you to explicitly resume with the `-r`/`--resume` argument.    

See [here](doc/convert.md) for more information.

### Examples

A number of example scripts are provided to showcase the features of the backend and generator. Some of them have hardcoded model paths and should be edited before you run them, but there is a simple CLI chatbot that you can start with:

```sh
python examples/chat.py -m <input_dir> -mode <prompt_mode> 

# E.g.:
python examples/chat.py -m /mnt/models/llama3.1-8b-instruct-exl3 -mode llama3
```

## EXL3 quantization

<div align="center">
    <a href="doc/exl3.md" target="_blank">
        <img src="doc/llama31_8b_instruct_bpw.png" width="640">
    </a>
</div>

Despite their amazing achievements, most SOTA quantization techniques remain cumbersome or even prohibitively expensive to use. For instance, **AQLM** quantization of a 70B model takes around **720 GPU-hours** on an A100 server, costing $850 US at the time of writing. ExLlamaV3 aims to address this with the **EXL3** format, which is a streamlined variant of [**QTIP**](https://github.com/Cornell-RelaxML/qtip) from Cornell RelaxML. The conversion process is designed to be simple and efficient and requires only an input model (in HF format) and a target bitrate. By computing Hessians on the fly and thanks to a fused Viterbi kernel, the quantizer can convert a model in a single step, taking a couple of minutes for smaller models, up to a few hours for larger ones (70B+) (on a single RTX 4090 or equivalent GPU.)

The [Marlin](https://github.com/IST-DASLab/marlin)-inspired GEMM kernel achieves roughly memory-bound latency under optimal conditions (4bpw, RTX 4090), though it still needs some work to achieve the same efficiency on Ampere GPUs and to remain memory-bound at lower bitrates.

Since converted models largely retain the original file structure (unlike **EXL2** which renames some tensors in its quest to turn every model into a Llama variant), it will be possible to extend **EXL3** support to other frameworks like HF Transformers and vLLM.

There are some benchmark results [here](doc/exl3.md), and a full writeup on the format is coming soon.

Fun fact: Llama-3.1-70B-EXL3 is coherent at 1.6 bpw. With the output layer quantized to 3 bpw and a 4096-token cache, inference is possible in under 16 GB of VRAM. 

A selection of EXL3-quantized models is available on [🤗 Hugging Face](https://huggingface.co/collections/turboderp/exl3-models-67f2dfe530f05cb9f596d21a).


## Acknowledgements

This project owes its existence to a wonderful community of FOSS developers and some very generous supporters (🐈❤️!) The following projects in particular deserve a special mention:

- [TabbyAPI](https://github.com/theroyallab/tabbyAPI/)
- [PyTorch](https://github.com/pytorch/pytorch)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [QTIP](https://github.com/Cornell-RelaxML/qtip)
- [Transformers](https://github.com/huggingface/transformers)
- [Marlin](https://github.com/IST-DASLab/marlin)
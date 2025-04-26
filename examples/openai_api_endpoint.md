# exllamav3 OpenAI-Compatible API Endpoint

This document describes the usage of the `openai_api_endpoint.py` script, which provides an OpenAI-compatible API endpoint for interacting with the exllamav3 server.

## Purpose

The script launches a local HTTP server that mimics the OpenAI API structure, specifically the `/v1/chat/completions` and `/v1/models` endpoints. This allows applications designed to work with the OpenAI API to interact with locally running exllamav3 models with minimal changes.

## Running the Script

To run the server, execute the script from your terminal:

```bash
python examples/openai_api_endpoint.py --model-dir /path/to/your/model [other_options]
```

Example:
```bash
python examples/openai_api_endpoint.py --model_dir QwQ-32B-exl3-4bpw/ --port 9999 --host 0.0.0.0 --log-level DEBUG --cache_size 16384 --cache_quant 4
 ```

### Command-Line Arguments

The script accepts several arguments:

*   **`--model-dir` (Required)**: Path to the directory containing the ExLlamaV3 model files. This is passed to the underlying `model_init` function.
*   **`-p`, `--port`**: The port number the server should listen on. (Default: `8000`)
*   **`-H`, `--host`**: The hostname or IP address the server should bind to. (Default: `localhost`)
*   **`--log-level`**: Sets the logging level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. (Default: `INFO`)

**Note:** The script also accepts all arguments supported by `exllamav3.model_init.add_args()`, which are used for model initialization (e.g., `--gpu_split`, `--length`, `--batch_size`, `--cache_size`, `--cache_quant`, --etc.). Refer to the `model_init` documentation or source for a full list.

## API Endpoints

### `GET /v1/models`

Lists the model currently loaded by the server.

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "model_name", // Derived from the --model-dir path
      "object": "model",
      "created": 1677609600, // Example timestamp
      "owned_by": "user",
      "permission": [],
      "root": "model_name",
      "parent": null
    }
  ]
}
```

**Example (`curl`):**

```bash
curl http://localhost:8000/v1/models
```

### `POST /v1/chat/completions`

Generates chat completions based on a provided list of messages. Supports both streaming and non-streaming responses.

**Request Body (JSON):**

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "model": "model_name", // Optional, defaults to the loaded model name
  "stream": false,       // Set to true for streaming responses (default: true)
  "max_tokens": 200,     // Max tokens to generate (default: 200)
  "temperature": 0.8,    // Sampling temperature (default: 0.8)
  "top_p": 0.8,          // Nucleus sampling p (default: 0.8)
  "top_k": 50,           // Top-k sampling k (default: 50)
  "stop": []             // Optional list of stop strings or token IDs
}
```

**Non-Streaming Response (JSON):**

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1677609600,
  "model": "model_name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hi there! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 10,
    "total_tokens": 25
  }
}
```

**Streaming Response (Server-Sent Events):**

The server sends a stream of JSON objects prefixed with `data: `. The stream ends with `data: [DONE]`.

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi "},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"there!"},"finish_reason":null}]}

...
data: [DONE]

```

**Examples (`curl`):**

*   **Non-Streaming:**

    ```bash
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [{"role": "user", "content": "Write a short poem about code."}],
        "stream": false,
        "max_tokens": 50
      }'
    ```

*   **Streaming:**

    ```bash
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [{"role": "user", "content": "Explain quantum entanglement simply."}],
        "stream": true,
        "max_tokens": 100
      }'
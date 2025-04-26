import argparse
import json
import time
import uuid
import sys
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logger
logger: logging.Logger = logging.getLogger(__name__)

# Add project root to path
script_path: str = os.path.abspath(__file__)
project_root: str = os.path.dirname(os.path.dirname(script_path))
sys.path.insert(0, project_root)

try:
    # Corrected imports based on package structure
    from exllamav3 import model_init # Import the initialization helper
    from exllamav3.models.config import Config # Keep for potential type hinting or direct use if needed elsewhere
    from exllamav3.models.model import Model # Keep for potential type hinting
    from exllamav3.cache.cache import Cache # Keep for potential type hinting
    from exllamav3.tokenizer.tokenizer import Tokenizer # Keep for potential type hinting
    from exllamav3.generator.generator import Generator # Import the base Generator class
    from exllamav3.generator.sampler.sampler import Sampler # Base sampler class
    from exllamav3.generator.sampler.custom import CustomSampler, SS_Temperature, SS_TopP, SS_TopK, SS_Sample # Custom sampler components
    from exllamav3.generator.job import Job # Import Job class
    import torch
except ImportError as e:
    logger.error(f"Error: Required libraries not found. {e}") # Changed to logger.error
    logger.error("Please ensure you have PyTorch installed and the exllamav3 library is accessible.") # Changed to logger.error
    sys.exit(1)

# Global variables for model, tokenizer, generator
model: Optional[Model] = None
tokenizer: Optional[Tokenizer] = None
generator: Optional[Generator] = None
cache: Optional[Cache] = None # Added global cache variable
model_name: str = "exllamav3_model" # Default model name, can be overridden by request

def initialize_model(args: argparse.Namespace) -> None:
    global model, tokenizer, generator, model_name, cache # Add cache to globals

    # Use the model_init helper function for robust initialization
    logger.info(f"Initializing model: {args.model_dir}")
    initialized_model, config, initialized_cache, initialized_tokenizer = model_init.init(args, quiet = True) # Capture the returned cache
    model = initialized_model
    cache = initialized_cache
    tokenizer = initialized_tokenizer


    logger.info("Model loaded.")
    logger.info("Tokenizer loaded.")
    # The cache is now created and allocated by model_init.init
    if cache:
        logger.info(f"Cache initialized by model_init with max_num_tokens = {cache.max_num_tokens}")
    else:
        logger.warning("Cache was not initialized by model_init.") # Should not happen if cache=True

    # Generator uses the cache returned by model_init.init
    if model and cache and tokenizer:
        generator = Generator(model, cache, tokenizer)
        logger.info("Generator created.")
    else:
        logger.error("Failed to initialize generator due to missing model, cache, or tokenizer.")
        # Potentially raise an error or exit if generator is critical
        sys.exit(1)


    # Use directory name as a default model identifier
    model_name = os.path.basename(os.path.normpath(args.model_dir))

    logger.info("Initialization complete.")

def format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Formats a list of messages into a single prompt string using the tokenizer's
    chat template, if available, or a basic format otherwise.
    """
    global tokenizer # Ensure tokenizer is accessible
    if tokenizer and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        # Use the tokenizer's apply_chat_template method if available
        try:
            # Ensure messages are in the format expected by apply_chat_template
            # (typically list of dicts with 'role' and 'content')
            formatted_prompt: str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return formatted_prompt
        except Exception as e:
            logger.warning(f"Tokenizer has chat_template but failed to apply: {e}. Falling back to basic formatting.")
            # Fallback if template application fails
            prompt: str = ""
            for msg in messages:
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "assistant:" # Basic prompt for generation
            return prompt
    else:
        # Basic fallback formatting
        prompt: str = ""
        for msg in messages:
            prompt += f"{msg['role']}: {msg['content']}\n"
        prompt += "assistant:" # Basic prompt for generation
        return prompt

class APIHandler(BaseHTTPRequestHandler):
    def _send_response(self, status_code: int, content_type: str, body: str) -> None:
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*') # Basic CORS
        self.end_headers()
        self.wfile.write(body.encode('utf-8'))

    def _send_json_response(self, status_code: int, data: Dict[str, Any]) -> None:
        self._send_response(status_code, 'application/json', json.dumps(data))

    def _send_error(self, status_code: int, message: str) -> None:
        error_data: Dict[str, Any] = {"error": {"message": message, "type": "api_error", "code": None}}
        self._send_json_response(status_code, error_data)

    def _send_sse_chunk(self, data: Union[Dict[str, Any], str]) -> bool:
        """Sends a Server-Sent Event chunk."""
        try:
            if isinstance(data, str): # Handle "[DONE]" string
                 self.wfile.write(f"data: {data}\n\n".encode('utf-8'))
            else:
                 self.wfile.write(f"data: {json.dumps(data)}\n\n".encode('utf-8'))
            self.wfile.flush() # Ensure data is sent immediately
            return True
        except BrokenPipeError:
            return False # Stop streaming if client disconnects

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def do_POST(self) -> None:
        global model, tokenizer, generator, model_name, cache # Ensure globals are accessible

        if self.path == '/v1/chat/completions':
            try:
                content_length_header = self.headers['Content-Length']
                if content_length_header is None:
                    self._send_error(411, "Content-Length header is required.")
                    return
                content_length: int = int(content_length_header)
                post_data: bytes = self.rfile.read(content_length)
                request_data: Dict[str, Any] = json.loads(post_data.decode('utf-8'))
            except (TypeError, ValueError, json.JSONDecodeError) as e:
                self._send_error(400, f"Invalid JSON request: {e}")
                return
            except Exception as e:
                 self._send_error(400, f"Error reading request: {e}")
                 return

            if not model or not tokenizer or not generator or not cache:
                 self._send_error(503, "Model is not initialized yet.")
                 return

            messages: Optional[List[Dict[str, str]]] = request_data.get('messages')
            if not messages or not isinstance(messages, list):
                self._send_error(400, "'messages' field is required and must be a list.")
                return

            # --- Parameters ---
            stream: bool = request_data.get('stream', True)
            max_tokens: int = request_data.get('max_tokens', 200) # Default max tokens
            temperature: float = request_data.get('temperature', 0.8)
            top_p: float = request_data.get('top_p', 0.8)
            top_k: int = request_data.get('top_k', 50)
            # Note: Add other sampler settings as needed (rep penalty, frequency/presence, etc.)
            requested_model_name: str = request_data.get('model', model_name) # Use loaded model name if not specified

            # --- Prepare Prompt ---
            try:
                prompt: str = format_chat_prompt(messages)
                # Encode prompt to get input IDs
                input_ids: torch.Tensor = tokenizer.encode(prompt)
                prompt_token_count: int = input_ids.shape[-1]
            except Exception as e:
                self._send_error(500, f"Error processing messages or encoding prompt: {e}")
                return

            # --- Check Token Limits ---
            required_tokens: int = prompt_token_count + max_tokens
            capacity: int = cache.max_num_tokens # Use the initialized cache object
            if required_tokens > capacity:
                logger.warning(f"Request rejected: Required tokens ({required_tokens} = {prompt_token_count} prompt + {max_tokens} max_new) exceed model capacity ({capacity}).")
                error_payload: Dict[str, Any] = {
                    "error": {
                        "message": f"Requested tokens ({required_tokens}) exceed model capacity ({capacity}). Prompt tokens: {prompt_token_count}, max_new_tokens: {max_tokens}.",
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded" # Added a potential code
                    }
                }
                # Use _send_json_response for consistency
                self._send_json_response(400, error_payload)
                return
            # --- Generation Settings ---
            # Instantiate a custom sampler with the requested parameters
            sampler_steps: List[Any] = [] # List of sampler setting instances
            if temperature > 0.01: # Avoid division by zero or near-zero
                sampler_steps.append(SS_Temperature(temperature))
            if top_k > 1:
                 sampler_steps.append(SS_TopK(top_k))
            if 0 < top_p < 1.0:
                 sampler_steps.append(SS_TopP(top_p))
            # Add the final sampling step (e.g., SS_Sample for random sampling)
            sampler_steps.append(SS_Sample()) # Or SS_Argmax() for greedy

            sampler: CustomSampler = CustomSampler(sampler_steps)
            # Note: Repetition penalty, etc., would need corresponding SS_ steps added if supported/desired.
            # Note: Disallowing tokens needs to be handled differently, likely via Job parameters or sampler logic.

            generation_id: str = f"chatcmpl-{uuid.uuid4()}"
            created_time: int = int(time.time())

            # --- Streaming Response ---
            if stream:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                finish_reason: Optional[str] = None
                job_done: bool = False
                job: Optional[Job] = None # Initialize job to None
                try:
                    # Define stop conditions based on request
                    stop_conditions_req: Any = request_data.get('stop', [])
                    stop_conditions: List[Union[str, int]] = []
                    if isinstance(stop_conditions_req, list):
                        stop_conditions = stop_conditions_req
                    elif isinstance(stop_conditions_req, (str, int)):
                         stop_conditions = [stop_conditions_req]
                    # Add common stop tokens if necessary, e.g., tokenizer.eos_token_id
                    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop_conditions:
                         stop_conditions.append(tokenizer.eos_token_id)


                    # Create and enqueue the job
                    job = Job(
                        input_ids=input_ids,
                        max_new_tokens=max_tokens,
                        sampler=sampler,
                        stop_conditions=stop_conditions,
                        decode_special_tokens=True # Assuming we want special tokens decoded
                    )
                    generator.enqueue(job)

                    # Iterate while the job is active
                    while generator.num_remaining_jobs() > 0 and not job_done:
                        results: List[Dict[str, Any]] = generator.iterate()
                        for r in results:
                            # Ensure we are processing results for our specific job
                            if job and r["serial"] != job.serial_number:
                                continue

                            if r["stage"] == "streaming":
                                chunk: str = r.get("text", "")
                                if chunk: # Only send if there's text
                                    stream_chunk: Dict[str, Any] = {
                                        "id": generation_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": requested_model_name,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"role": "assistant", "content": chunk},
                                            "finish_reason": None
                                        }]
                                    }
                                    if not self._send_sse_chunk(stream_chunk):
                                        logger.warning("Client disconnected, cancelling job.") # Changed to logger.warning
                                        if job: generator.cancel(job) # Cancel the job if client disconnects
                                        job_done = True
                                        break # Exit inner results loop

                                if r["eos"]:
                                    finish_reason = r.get("eos_reason", "stop")
                                    job_done = True
                                    break # Exit inner results loop
                        if job_done:
                            break # Exit outer while loop

                    # Send final chunk after the loop finishes or breaks
                    final_chunk: Dict[str, Any] = {
                        "id": generation_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": requested_model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason if finish_reason else "stop" # Provide default if None
                        }],
                        # Usage info could be added here if tracked/available from Job results
                    }
                    self._send_sse_chunk(final_chunk)
                    self._send_sse_chunk("[DONE]")

                except Exception as e:
                    logger.exception(f"Error during streaming generation: {e}") # Use exception to include traceback
                    # Try to send an error message if connection is still open
                    try:
                        error_chunk: Dict[str, Any] = {"error": {"message": f"Generation error: {e}", "type": "generation_error"}}
                        self._send_sse_chunk(error_chunk)
                        self._send_sse_chunk("[DONE]") # Still send DONE marker
                    except:
                        pass # Ignore errors if we can't even send the error message
                    finally:
                        # Ensure job is cancelled if an error occurred during streaming setup/loop
                        if job and not job_done:
                            generator.cancel(job)


            # --- Non-Streaming Response ---
            else:
                try:
                    # Define stop conditions based on request (example, adapt as needed)
                    stop_conditions_req: Any = request_data.get('stop', [])
                    stop_conditions: List[Union[str, int]] = []
                    if isinstance(stop_conditions_req, list):
                        stop_conditions = stop_conditions_req
                    elif isinstance(stop_conditions_req, (str, int)):
                         stop_conditions = [stop_conditions_req]
                    # Add common stop tokens if necessary, e.g., tokenizer.eos_token_id
                    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop_conditions:
                         stop_conditions.append(tokenizer.eos_token_id)


                    # Use the generator.generate utility function
                    completion: str
                    last_results: Optional[Dict[str, Any]]
                    completion, last_results = generator.generate(
                        prompt=prompt, # Ensure the raw prompt string is passed
                        max_new_tokens=max_tokens,
                        sampler=sampler, # Pass the created sampler instance
                        stop_conditions=stop_conditions,
                        return_last_results=True,
                        decode_special_tokens=True # Assuming we want special tokens decoded for OpenAI format
                    )

                    # Extract necessary info from the results
                    full_response_text: str = completion # Assuming generate returns only completion here
                    finish_reason: str = last_results.get("eos_reason", "stop") if last_results else "stop"
                    completion_token_count: int = last_results.get("new_tokens", 0) if last_results else 0

                    response_data: Dict[str, Any] = {
                        "id": generation_id,
                        "object": "chat.completion",
                        "created": created_time,
                        "model": requested_model_name,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": full_response_text.strip() # Strip potential whitespace
                            },
                            "finish_reason": finish_reason
                        }],
                        "usage": {
                            "prompt_tokens": prompt_token_count,
                            "completion_tokens": completion_token_count, # This is an approximation
                            "total_tokens": prompt_token_count + completion_token_count
                        }
                    }
                    self._send_json_response(200, response_data)

                except Exception as e:
                    logger.exception(f"Error during non-streaming generation: {e}") # Use exception
                    self._send_error(500, f"Error during generation: {e}")

        else:
            self._send_error(404, "Not Found. Use POST /v1/chat/completions")

    def do_GET(self) -> None:
        global model, tokenizer, generator, model_name # Ensure globals are accessible

        if self.path == '/v1/models':
             if not model or not tokenizer or not generator:
                 self._send_error(503, "Model is not initialized yet.")
                 return
             # Provide basic model info
             model_info: Dict[str, Any] = {
                 "object": "list",
                 "data": [
                     {
                         "id": model_name,
                         "object": "model",
                         "created": int(time.time()), # Placeholder time
                         "owned_by": "user", # Placeholder owner
                         "permission": [],
                         "root": model_name,
                         "parent": None,
                     }
                 ]
             }
             self._send_json_response(200, model_info)
        else:
            self._send_response(200, 'text/plain', 'ExLlamaV3 API Server is running. Use POST /v1/chat/completions.')

    # Override default request logging to use our logger
    def log_message(self, format: str, *args: Any) -> None:
        logger.info(format % args) # Use logger.info or logger.debug as preferred


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass

if __name__ == "__main__":
    # Argument parsing moved here
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="ExLlamaV3 OpenAI-Compatible API Endpoint")
    # Add args from model_init FIRST - crucial for model_init.init() to work correctly
    model_init.add_args(parser, cache=True)
    # Specific args for this script
    parser.add_argument("-p", "--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("-H", "--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    args: argparse.Namespace = parser.parse_args()

    # Configure logging
    log_level_str: str = args.log_level.upper()
    log_level: int = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s:%(funcName)-16s - %(levelname)-8s - %(message)s',
                        force=True) # Use force=True to allow reconfiguration

    # Ensure essential args for model_init have defaults if not provided
    if not hasattr(args, 'gpu_split'):
        args.gpu_split = None # Set default if missing

    # Initialize model in the main thread (or consider a separate setup phase)
    try:
        initialize_model(args)
    except Exception as e:
        logger.exception(f"Fatal error during model initialization: {e}") # Use exception
        sys.exit(1)

    server_address: Tuple[str, int] = (args.host, args.port)
    httpd: ThreadingHTTPServer = ThreadingHTTPServer(server_address, APIHandler)

    logger.info(f"Starting OpenAI-compatible server on http://{args.host}:{args.port}...")
    logger.info(f"Using model: {model_name} from {args.model_dir}")
    logger.info("Endpoint available at POST /v1/chat/completions")
    logger.info("Model list available at GET /v1/models")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.exception(f"Server encountered an unexpected error: {e}")
    finally:
        logger.info("Closing server...")
        httpd.server_close()
        logger.info("Server closed.")
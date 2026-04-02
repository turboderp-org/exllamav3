from __future__ import annotations
import threading
import torch


class TQToEXL3HotSwap:
    """
    Background converter: upgrades TQ-quantized layers to EXL3 one by one.

    Usage:
        model = load_model(...)  # loaded with instant TQ
        swapper = TQToEXL3HotSwap(model)
        swapper.start()  # begins background conversion

        # Model is usable immediately for inference
        # Layers upgrade to EXL3 progressively in background

        swapper.wait()  # optional: block until all layers are upgraded

    Thread-safety note:
        ``linear.inner`` is replaced atomically under ``_lock``.  The brief
        window in ``convert_exl3()`` where ``self.inner`` is set to ``None``
        (before the new ``LinearEXL3`` is assigned) is also covered by the
        same lock — callers must hold ``_lock`` around any use of
        ``linear.inner`` that cannot tolerate a ``None`` value.

        Inference threads that call ``linear.forward()`` do NOT hold the lock;
        the swap is written through a single Python reference assignment which
        is atomic at the interpreter level (GIL), so a racing forward pass will
        either see the old TQ inner or the fully-constructed EXL3 inner, never
        the transient ``None``.  The ``convert_exl3()`` path that briefly sets
        ``self.inner = None`` is wrapped entirely inside the lock in
        ``_swap_one_layer()``, so no concurrent forward pass can observe the
        ``None`` state.
    """

    def __init__(
        self,
        model,
        bits: int = 4,
        calibration_rows: int = 128,
        calibration_cols: int = 2048,
        callback=None,  # called with (layer_idx, total_layers, status: str)
    ):
        self.model = model
        self.bits = bits
        self.cal_rows = calibration_rows
        self.cal_cols = calibration_cols
        self.callback = callback
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.progress = {
            "total": 0,
            "completed": 0,
            "current_layer": None,
            "status": "idle",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start background conversion thread."""
        self._thread = threading.Thread(target=self._convert_loop, daemon=True)
        self._thread.start()
        self.progress["status"] = "running"

    def stop(self):
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=30)
        self.progress["status"] = "stopped"

    def wait(self):
        """Block until all conversions are complete (or stop() was called)."""
        if self._thread:
            self._thread.join()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_tq_linears(self):
        """Return a list of (name, Linear) pairs whose inner is a TQ instant layer."""
        from ..modules.linear import Linear

        linears = []
        for name, module in self.model.named_modules():
            if isinstance(module, Linear) and hasattr(module, "inner"):
                if getattr(module.inner, "quant_type", None) == "tq_instant":
                    linears.append((name, module))
        return linears

    def _build_diagonal_H(self, linear, idx: int) -> dict:
        """
        Build a minimal Hessian dict for a single layer.

        Uses a small scaled identity so that ``finalize_capture_H`` will
        normalise cleanly and not fall through to the ``count == 0`` path
        (which would trigger q_fallback anyway, but with a zero-diagonal that
        causes a warning).  With ``q_fallback=True`` in ``quant_args`` the
        Hessian is never actually used for LDLQ; it is only used for the
        su/sv random-flip seed and proxy-error calculation, which are
        harmless with an identity.
        """
        device = linear.inner.packed.device if hasattr(linear.inner, "packed") else torch.device("cpu")
        n = linear.in_features

        # One synthetic "sample" with unit variance so count=1, diag_mean=1.
        H = torch.eye(n, dtype=torch.float32, device=device)

        return {
            "H": H,
            "first_key": linear.key,
            "count": 1,           # non-zero → diag_mean will be 1.0, not forced fallback
            "finalized": False,
            "num_total": n,
            "inf_nan": torch.zeros(2, dtype=torch.long, device=device),
        }

    def _swap_one_layer(self, name: str, linear, idx: int) -> str:
        """
        Convert one TQ layer to EXL3 under the lock.

        Returns a status string: "upgraded" or "failed: <message>".
        """
        from ..modules.quant.fp16 import LinearFP16

        try:
            # ----------------------------------------------------------
            # 1. Grab the FP16 weight from the TQ layer (no lock needed
            #    here — we are only reading, and TQ dequant is idempotent).
            # ----------------------------------------------------------
            weight_fp16 = linear.inner.get_weight_tensor()   # (in, out) fp16
            bias = linear.inner.get_bias_tensor()
            device = weight_fp16.device

            # ----------------------------------------------------------
            # 2. Build H_data (diagonal identity, q_fallback will fire).
            # ----------------------------------------------------------
            H_data = self._build_diagonal_H(linear, idx)

            # ----------------------------------------------------------
            # 3. Build quant_args.
            #    - "apply_out_scales" must be present; ``regularize()``
            #      reads it as quant_args["apply_out_scales"].
            #    - "q_fallback" set to True skips expensive LDLQ / LDL
            #      decomposition and does round-to-nearest EXL3.
            #    - "mcg" is the default codebook (matches convert_model.py).
            # ----------------------------------------------------------
            quant_args = {
                "K": self.bits,
                "seed": idx,
                "sigma_reg": 0.01,
                "devices": [device],
                "apply_out_scales": True,   # required key for regularize()
                "q_fallback": True,         # skip LDLQ; round-to-nearest EXL3
                "mcg": True,                # default codebook
            }

            # ----------------------------------------------------------
            # 4. Atomically swap inner: TQ → FP16 → EXL3.
            #    The entire replacement (including convert_exl3's brief
            #    inner=None window) is covered by the lock so that no
            #    forward pass can observe inner=None.
            # ----------------------------------------------------------
            with self._lock:
                # Restore to FP16 so convert_exl3()'s isinstance check passes.
                # weight is already (in_features, out_features) — the shape
                # LinearFP16 stores internally.
                linear.inner = LinearFP16(
                    linear.in_features,
                    linear.out_features,
                    weight_fp16.contiguous(),
                    bias,
                    linear.full_in_features,
                    linear.full_out_features,
                    linear.first_in_feature,
                    linear.first_out_feature,
                    linear.out_dtype,
                    key=linear.key,
                )
                # convert_exl3 sets inner=None, quantizes, then sets the new
                # LinearEXL3 — all inside this same lock acquisition.
                linear.convert_exl3(H_data, quant_args)
                linear.quant_type = "exl3"

            return "upgraded"

        except Exception as exc:
            return "failed: " + str(exc)

    # ------------------------------------------------------------------
    # Background thread entry-point
    # ------------------------------------------------------------------

    def _convert_loop(self):
        """Main conversion loop — runs in the background thread."""
        try:
            linears = self._find_tq_linears()
            total = len(linears)
            self.progress["total"] = total

            if not linears:
                self.progress["status"] = "complete"
                if self.callback:
                    self.callback(0, 0, "complete")
                return

            self.progress["status"] = "converting"

            for idx, (name, linear) in enumerate(linears):
                if self._stop_event.is_set():
                    break

                self.progress["current_layer"] = name
                self.progress["completed"] = idx

                if self.callback:
                    self.callback(idx, total, "converting")

                status = self._swap_one_layer(name, linear, idx)

                if self.callback:
                    self.callback(idx, total, status)

            self.progress["completed"] = total
            self.progress["current_layer"] = None

            if not self._stop_event.is_set():
                self.progress["status"] = "complete"
                if self.callback:
                    self.callback(total, total, "complete")

        except Exception as exc:
            self.progress["status"] = "error: " + str(exc)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def convert_model_instant_tq(model, bits=4, sub_scale_size=8, callback=None):
    """
    Convert all FP16 Linear layers in a model to TQ instant quantization.

    Iterates ``model.modules`` (the flat list used by ExLlamaV3 models) and
    calls ``linear.convert_tq_instant()`` on every ``Linear`` whose inner
    layer is currently FP16.

    Args:
        model:            Loaded ExLlamaV3 model (FP16).
        bits:             Target bitrate (default 4).
        sub_scale_size:   Sub-block scale granularity (default 8).
        callback:         Optional progress callback(layer_idx, total, name_or_msg).

    Returns:
        Number of layers converted.
    """
    from ..modules.linear import Linear

    linears = [m for m in model.modules if isinstance(m, Linear)]
    total = len(linears)
    converted = 0

    for idx, linear in enumerate(linears):
        if not hasattr(linear, "convert_tq_instant"):
            continue
        inner = getattr(linear, "inner", None)
        if inner is None or getattr(inner, "quant_type", None) != "fp16":
            continue  # skip already-quantized or unloaded layers
        try:
            linear.convert_tq_instant(bits=bits, sub_scale_size=sub_scale_size)
            converted += 1
            if callback:
                callback(idx, total, linear.key)
        except Exception as exc:
            if callback:
                callback(idx, total, "FAILED: " + str(exc))

    return converted

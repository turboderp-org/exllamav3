"""SenseNova Loop's FP32 router (distinct from DeepSeek hyperconnections)."""
import torch
import torch.nn.functional as F
from .module import Module


def route(residual, weight, bias, streams, iterations, fixed_identity, pre_scale):
    with torch.autocast(device_type = residual.device.type, enabled = False):
        flat = residual.flatten(-2).float()
        normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim = True) + 1e-6)
        logits = F.linear(normalized, weight, bias)
        pre = logits[..., :streams].sigmoid()
        post = 2 * logits[..., streams:2 * streams].sigmoid()
        if fixed_identity:
            mix = torch.eye(streams, device = flat.device, dtype = torch.float32).expand(
                *flat.shape[:-1], streams, streams)
        else:
            mix = logits[..., 2 * streams:].unflatten(-1, (streams, streams)).float()
            for _ in range(iterations):
                mix = mix - torch.logsumexp(mix, dim = -1, keepdim = True)
                mix = mix - torch.logsumexp(mix, dim = -2, keepdim = True)
            mix = mix.exp()
        branch = torch.matmul(pre.to(residual.dtype).unsqueeze(-2), residual).squeeze(-2)
        if pre_scale != 1.0:
            branch = branch * pre_scale
        return branch, post, mix


def merge(residual, branch, post, mix):
    with torch.autocast(device_type = residual.device.type, enabled = False):
        return torch.matmul(mix.to(residual.dtype), residual) + post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)


class LoopRouter(Module):
    def __init__(self, config, key, settings, hidden_size):
        super().__init__(config, key, None)
        self.settings = settings
        self.hidden_size = hidden_size
        self.weight = self.bias = None

    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        self.weight = self.config.stc.get_tensor(self.key + ".weight", device, no_defer = True, arena = False).float().contiguous()
        self.bias = self.config.stc.get_tensor(self.key + ".bias", device, no_defer = True, arena = False).float().contiguous()

    def unload(self):
        super().unload()
        self.weight = self.bias = None

    def get_tensors(self):
        return {self.key + ".weight": self.weight.contiguous(), self.key + ".bias": self.bias.contiguous()}

    def weights_numel(self):
        s = self.settings["streams"]
        return (2 * s + (0 if self.settings["fixed_identity"] else s * s)) * (s * self.hidden_size + 1)

    def optimizer_targets(self):
        return []

    def forward(self, x, params, out_dtype = None):
        s = self.settings
        return route(x, self.weight, self.bias, s["streams"], s["iterations"], s["fixed_identity"], s.get("pre_scale", 1.0))

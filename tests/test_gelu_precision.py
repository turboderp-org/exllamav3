"""GELU must retain FP32 accuracy over the complete finite FP16 input range."""
import torch
import torch.nn.functional as F
from exllamav3.ext import exllamav3_ext as ext

@torch.inference_mode()
def test_gelu_all_finite_half_inputs():
    x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.half)
    x=x[torch.isfinite(x)].cuda()
    y=torch.full_like(x,.75)
    want=(F.gelu(x.float(),approximate='tanh').half()*y).clamp(-65504,65504)
    got=torch.empty_like(x)
    ext.gelu_mul(x,y,got,0.)
    # tanh.approx plus two FP16 roundings can differ from PyTorch's tanh
    # implementation by two output ULPs; near zero use an absolute floor.
    spacing=(torch.nextafter(want,torch.full_like(want,float('inf')))-want).abs().float()
    assert ((got.float()-want.float()).abs() <= 2*spacing+1e-5).all()
    # The FP32 input path also returns FP16 products, but does not round the
    # activation before multiplication. Include the negative saturation region.
    xf=torch.linspace(-10,10,32768,device='cuda',dtype=torch.float)
    yf=torch.full_like(xf,.75)
    out=torch.empty_like(xf,dtype=torch.half)
    ext.gelu_mul(xf,yf,out,0.)
    ref=(F.gelu(xf,approximate='tanh')*yf).half()
    torch.testing.assert_close(out,ref,rtol=.001,atol=.0001)

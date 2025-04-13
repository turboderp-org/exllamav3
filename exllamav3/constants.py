# Fixed page size for generator
PAGE_SIZE = 256

# Maximum MLP size we can realistically quantize on one GPU. Wider MLP layers are split while quantizing
MAX_MLP_INTERMEDIATE = 55296
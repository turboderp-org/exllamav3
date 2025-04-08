import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.conversion.quant_config import create_quantization_config_json
import argparse

def main(args):
    filename = os.path.join(args.model_dir, "quantization_config.json")
    update = os.path.exists(filename)
    create_quantization_config_json(args.model_dir)
    if update:
        print(f"Updated {filename}")
    else:
        print(f"Created {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory", required = True)
    _args = parser.parse_args()
    main(_args)
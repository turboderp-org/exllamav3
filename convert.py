from exllamav3.conversion.convert_model import parser, main, prepare

# Script included in package: ./exllamav3/conversion/convert_model.py

if __name__ == "__main__":
    _args = parser.parse_args()
    _in_args, _job_state, _ok, _err = prepare(_args)
    if not _ok:
        print(f" !! Error: {_err}")
    else:
        main(_in_args, _job_state)
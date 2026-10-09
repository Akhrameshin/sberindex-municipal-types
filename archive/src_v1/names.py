import yaml
_N = yaml.safe_load(open("configs/type_names.yaml"))
def names(K): return {int(k): v for k, v in _N[f"K{K}"].items()}

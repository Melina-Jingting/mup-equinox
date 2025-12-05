# mup-equinox

## Coordinate Checks

You can run coordinate checks for different models using the provided scripts.
The scripts support command line arguments for specifying steps and optimizers.

**Syntax:**
```bash
python path/to/coord_check.py --steps <step1> <step2> ... --optimiser <opt1> <opt2> ...
```

**Examples:**
Run MLP coordinate check for steps 1, 10, and 100 with Adam and SGD:
```bash
python examples/mlp/coord_check.py --steps 1 10 100 --optimiser adam sgd
```
Run CNN coordinate check with default settings:
```bash
python examples/cnn/coord_check.py
```
Run SSM coordinate check with a name prefix:
```bash
python examples/ssm/coord_check.py --steps 200 --optimiser sgd --name_prefix "experiment_1_"
```

# Loss function

This script guides to reproduce our experiments regarding the effects of loss function on attention sink in LMs.


## Weight decay

```shell
bash scripts/loss_function/run_decay.sh $decay
```

You can choose `$decay` from 0, 1e-2, 1e-3, 1e0, 1e1, 2e0. 5e0, 5e-1.


## Prefix language modeling

```shell
bash scripts/loss_function/run_prefix.sh $prefix
```

You can choose `$prefix` from 2, 3, 4, 5, 10.

## Shifted window attention 

```shell
bash scripts/loss_function/run_window.sh $window
```

You can choose `$window` from 32, 64, 128, 256, 512, 1024.
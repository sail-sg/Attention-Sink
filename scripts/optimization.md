# Optimization

This script guides to reproduce our experiments regarding the effects of optimization on attention sink in LMs.


## Learning rate

```shell
bash scripts/optimization/run_lr.sh $lr
```

You can choose `$lr` from 1e-3, 1e-4 or 1e-5.

## Batch size

```shell
bash scripts/optimization/run_batch.sh $batch
```

You can choose `$batch` from 128, 256, 1024.
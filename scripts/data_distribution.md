# Data distribution

This script guides to reproduce our experiments regarding the effects of data distribution on attention sink in LMs.


## Data amount

```shell
bash scripts/data_distribution/run_data_amount.sh $downsample_ratio
```

You can choose `$downsample_ratio` yourself. `$downsample_ratio=1.0` means that total training data adds up to 5B tokens while `$downsample_ratio=y` (0.0 < y <= 1.0) means that total training data adds up to 5B*y tokens.


## Randomness in data distribution

First, we need to prepare random data, run

```shell
python random_data.py --start_token $start_token --end_token $end_token
```

Here `$start_token` represents the start of random token (inclusive) and `$end_token` represents the end of random token (exclusive). For instance, `$start_token=0` and `$end_token=1` refer to that the first token is a random token.


Afterward, run the pre-training script

```shell
bash scripts/data_distribution/run_random_token.sh $start_token $end_token
```



## Fix token in a specific position


```shell
bash scripts/data_distribution/run_fix_token.sh $fix_position
```

In our experiments, we select `$fix_position` from 0, 1, 2 and fix the token as the padding token (token id 1).


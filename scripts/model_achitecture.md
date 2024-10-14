# Model achitecture

This script guides to reproduce our experiments regarding the effects of model architecture on attention sink in LMs.


## Positional embedding

To pre-train LLaMA models with NoPE/Absolute PE/Learnable PE, run
```shell
bash scripts/model_achitecture/run_pos.sh $pos
```

You can choose `$pos` from nope, absolute, learnable.

To pre-train LLaMA models with ALiBI/Relative PE, run

```shell
bash scripts/model_achitecture/run_relpos.sh $relpos
```

You can choose `$relpos` from alibi, relative.


## Feed-forward network


```shell
bash scripts/model_achitecture/run_ffn.sh $ffn
```

You can choose `$ffn` from relu, gelu, swish, reglu, geglu.


## Post layer normalization

```shell
bash scripts/model_achitecture/run_postln.sh
```

## Attention biases

```shell
bash scripts/model_achitecture/run_kv_bias.sh $bias_suffix
```

You can choose `$bias_suffix` from k_bias, k_head_bias, kv_bias, kv_head_bias, v_bias, v_head_bias.


## Attention operations

In this section, since we modify the attention operation, it is not compatible with flash attentin. Therefore, we implement the model forward by ourselves, which require more GPU memory. We use 8 GPUs as default to run the following scripts. You can modify the micro batch size to when using less GPUs.

### Similarity functions

```shell
bash scripts/model_achitecture/run_sim.sh $sim
```

You can choose `$sim` from elu_norm, elu_no_norm, sigmoid_norm, sigmoid_no_norm.

### Kernal functions

```shell
bash scripts/model_achitecture/run_kernel.sh $kernel
```

You can choose `$kernel` from elu_norm, elu_no_norm, linear_norm, linear_no_norm, mlp_norm, mlp_no_norm.
"""
This script is used to prepare random data
"""

import os
import struct
import yaml
import glob
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}


def code(dtype):
    for k in dtypes:
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes

def random_train_tokens(random_token_start=0, random_token_end=2):
    block_size = 2049
    assert random_token_end > random_token_start
    vocab_size = 50432
    
    data_yaml_file = "./configs/tinyllama_60m.yaml"
    with open(data_yaml_file, "r") as f:
        config = yaml.safe_load(f)
    
    if "train" in config:
        train_config = []
        for k, v in config["train"].items():
            train_config.append((k, float(v)))
        # update the config
        data_config = train_config

    data_dir = Path("datasets/lit_dataset_regmix")
    all_filenames = []
    
    for idx in range(len(data_config)):
        prefix = data_config[idx][0]
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}-*")))
        all_filenames.extend(filenames)
    
    print(all_filenames)
    print(len(all_filenames))
    np.random.seed(1024)
    new_data_folder = f"datasets/lit_dataset_regmix_random{random_token_start}_{random_token_end}"
    os.makedirs(new_data_folder, exist_ok=True)
    for filename in tqdm(all_filenames):
        with open(filename, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        n_blocks = chunk_size // block_size
        mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
        buffer = memoryview(mmap)
        all_arr = []
        for block_idx in range(n_blocks):
            elem_id = (block_idx % n_blocks) * block_size
            offset = np.dtype(dtype).itemsize * elem_id
            buffer_length = len(buffer)
            offset = max(0, min(offset, buffer_length - 1))
            arr = np.frombuffer(buffer, dtype=dtype, count=block_size, offset=offset)
            random_arr = np.random.randint(0, vocab_size, size=(random_token_end-random_token_start,), dtype=dtype)
            # print(random_arr)
            new_arr = np.concatenate([arr[:random_token_start], random_arr, arr[random_token_end:]], axis=0)
            all_arr.append(new_arr)
        
        all_arr = np.concatenate(all_arr, axis=0).astype(dtype)
        
        new_path = os.path.join(new_data_folder, os.path.basename(filename))
        # print(new_path)
        # save
        with open(new_path, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<B", code(np.uint16)))
            f.write(struct.pack("<Q", 2049*256))
            f.write(all_arr.tobytes(order="C"))


def random_valid_tokens(random_token_start=0, random_token_end=2):
    block_size = 2049
    assert random_token_end > random_token_start
    vocab_size = 50432
    
    data_yaml_file = "./configs/tinyllama_60m.yaml"
    with open(data_yaml_file, "r") as f:
        config = yaml.safe_load(f)
    
    if "valid" in config:
        train_config = []
        for k, v in config["valid"].items():
            train_config.append((k, float(v)))
        # update the config
        data_config = train_config

    data_dir = Path("datasets/lit_dataset_regmix")
    all_filenames = []
    
    for idx in range(len(data_config)):
        prefix = data_config[idx][0]
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}-*")))
        all_filenames.extend(filenames)
    
    print(all_filenames)
    print(len(all_filenames))
    np.random.seed(1024)
    new_data_folder = f"datasets/lit_dataset_regmix_random{random_token_start}_{random_token_end}"
    os.makedirs(new_data_folder, exist_ok=True)
    for filename in tqdm(all_filenames):
        with open(filename, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        n_blocks = chunk_size // block_size
        mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
        buffer = memoryview(mmap)
        all_arr = []
        for block_idx in range(n_blocks):
            elem_id = (block_idx % n_blocks) * block_size
            offset = np.dtype(dtype).itemsize * elem_id
            buffer_length = len(buffer)
            offset = max(0, min(offset, buffer_length - 1))
            arr = np.frombuffer(buffer, dtype=dtype, count=block_size, offset=offset)
            random_arr = np.random.randint(0, vocab_size, size=(random_token_end-random_token_start,), dtype=dtype)
            # print(random_arr)
            new_arr = np.concatenate([arr[:random_token_start], random_arr, arr[random_token_end:]], axis=0)
            all_arr.append(new_arr)
        
        all_arr = np.concatenate(all_arr, axis=0).astype(dtype)
        
        new_path = os.path.join(new_data_folder, os.path.basename(filename))
        # print(new_path)
        # save
        with open(new_path, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<B", code(np.uint16)))
            f.write(struct.pack("<Q", 2049*256))
            f.write(all_arr.tobytes(order="C"))


def random_train_tokens_sanity_check(random_token_start=0, random_token_end=2):
    block_size = 2049
    assert random_token_end > random_token_start
    vocab_size = 50432
    
    data_yaml_file = "./configs/tinyllama_60m.yaml"
    with open(data_yaml_file, "r") as f:
        config = yaml.safe_load(f)
    
    if "train" in config:
        train_config = []
        for k, v in config["train"].items():
            train_config.append((k, float(v)))
        # update the config
        data_config = train_config

    data_dir = Path("datasets/lit_dataset_regmix")
    all_filenames = []
    
    for idx in range(len(data_config)):
        prefix = data_config[idx][0]
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}-*")))
        all_filenames.extend(filenames)

    print(len(all_filenames))
    new_data_folder = f"datasets/lit_dataset_regmix_random{random_token_start}_{random_token_end}"
    # os.makedirs(new_data_folder, exist_ok=True)
    for filename in tqdm(all_filenames[:4]):
        new_path = os.path.join(new_data_folder, os.path.basename(filename))
        with open(new_path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        n_blocks = chunk_size // block_size
        mmap = np.memmap(new_path, mode="r", order="C", offset=HDR_SIZE)
        buffer = memoryview(mmap)
        for block_idx in range(4):
            elem_id = (block_idx % n_blocks) * block_size
            offset = np.dtype(dtype).itemsize * elem_id
            buffer_length = len(buffer)
            offset = max(0, min(offset, buffer_length - 1))
            arr = np.frombuffer(buffer, dtype=dtype, count=block_size, offset=offset)
            print(arr[:10])
            # break
        # break

    for filename in tqdm(all_filenames[:4]):
        with open(filename, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        n_blocks = chunk_size // block_size
        mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
        buffer = memoryview(mmap)
        for block_idx in range(4):
            elem_id = (block_idx % n_blocks) * block_size
            offset = np.dtype(dtype).itemsize * elem_id
            buffer_length = len(buffer)
            offset = max(0, min(offset, buffer_length - 1))
            arr = np.frombuffer(buffer, dtype=dtype, count=block_size, offset=offset)
            print(arr[:10])
            # break
        # break


def random_valid_tokens_sanity_check(random_token_start=0, random_token_end=2):
    block_size = 2049
    assert random_token_end > random_token_start
    vocab_size = 50432
    
    data_yaml_file = "./configs/tinyllama_60m.yaml"
    with open(data_yaml_file, "r") as f:
        config = yaml.safe_load(f)
    
    if "valid" in config:
        train_config = []
        for k, v in config["valid"].items():
            train_config.append((k, float(v)))
        # update the config
        data_config = train_config

    data_dir = Path("datasets/lit_dataset_regmix")
    all_filenames = []
    
    for idx in range(len(data_config)):
        prefix = data_config[idx][0]
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}-*")))
        all_filenames.extend(filenames)

    print(len(all_filenames))
    new_data_folder = f"datasets/lit_dataset_regmix_random{random_token_start}_{random_token_end}"
    # os.makedirs(new_data_folder, exist_ok=True)
    for filename in tqdm(all_filenames[:4]):
        new_path = os.path.join(new_data_folder, os.path.basename(filename))
        with open(new_path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        n_blocks = chunk_size // block_size
        mmap = np.memmap(new_path, mode="r", order="C", offset=HDR_SIZE)
        buffer = memoryview(mmap)
        for block_idx in range(4):
            elem_id = (block_idx % n_blocks) * block_size
            offset = np.dtype(dtype).itemsize * elem_id
            buffer_length = len(buffer)
            offset = max(0, min(offset, buffer_length - 1))
            arr = np.frombuffer(buffer, dtype=dtype, count=block_size, offset=offset)
            print(arr[:10])
            # break
        # break

    for filename in tqdm(all_filenames[:4]):
        with open(filename, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        n_blocks = chunk_size // block_size
        mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
        buffer = memoryview(mmap)
        for block_idx in range(4):
            elem_id = (block_idx % n_blocks) * block_size
            offset = np.dtype(dtype).itemsize * elem_id
            buffer_length = len(buffer)
            offset = max(0, min(offset, buffer_length - 1))
            arr = np.frombuffer(buffer, dtype=dtype, count=block_size, offset=offset)
            print(arr[:10])
            # break
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_token", type=int, default=0, help="the start of random token (inclusive)")
    parser.add_argument("--end_token", type=int, default=1, help="the end of random token (exclusive)")
    args = parser.parse_args()
    random_train_tokens(random_token_start=args.start_token, random_token_end=args.end_token)
    random_valid_tokens(random_token_start=args.start_token, random_token_end=args.end_token)

    # sanity check
    print("sanity check!")
    random_train_tokens_sanity_check(random_token_start=args.start_token, random_token_end=args.end_token)
    random_valid_tokens_sanity_check(random_token_start=args.start_token, random_token_end=args.end_token)
#!/usr/bin/env python3
# Copyright 2026
#
# Standalone HF -> Megatron-Core checkpoint converter for Qwen3.5-VL / Qwen3.5-MoE-VL
# using mbridge.
#
# Example:
#   PYTHONPATH=/data/yanguo.sun/qwen35_verl/mbridge:/data/yanguo.sun/qwen35_verl/open/verl:/data/yanguo.sun/qwen35_verl/Megatron-LM:/data/yanguo.sun/qwen35_verl/megatron-lm-musa-patch:$PYTHONPATH \
#   torchrun --nproc_per_node=1 open/verl/scripts/mbridge_convert_qwen3_5_hf_to_mcore.py \
#     --hf-model-path /path/to/qwen3.5-vl-hf \
#     --output-path /path/to/qwen3.5-vl-mcore \
#     --trust-remote-code

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoConfig


def patch_torch_jit_script_for_transformer_engine() -> None:
    """Let Transformer Engine import succeed on torch_musa builds.

    Some torch_musa/PyTorch combinations fail while importing transformer_engine
    because TE decorates helper functions with torch.jit.script and TorchScript
    tries to inspect a builtin_function_or_method. The converter does not execute
    TE attention kernels, but mbridge/Megatron may import TE modules while building
    specs. In that case, falling back to the original Python function is enough.
    """

    original_script = torch.jit.script

    if getattr(original_script, "_mbridge_te_safe_patch", False):
        return

    def safe_script(obj=None, *args, **kwargs):
        try:
            if obj is None:
                return original_script(*args, **kwargs)
            return original_script(obj, *args, **kwargs)
        except TypeError as exc:
            if "builtin_function_or_method" in str(exc) and obj is not None:
                return obj
            raise

    safe_script._mbridge_te_safe_patch = True
    torch.jit.script = safe_script


patch_torch_jit_script_for_transformer_engine()

from mbridge import AutoBridge  # noqa: E402,F401 - importing mbridge also registers model bridges
from mbridge.core.util import unwrap_model  # noqa: E402
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.5 HF checkpoint to Megatron-Core distributed checkpoint with mbridge."
    )
    parser.add_argument("--hf-model-path", required=True, help="Input HuggingFace model directory.")
    parser.add_argument("--output-path", required=True, help="Output Megatron-Core distributed checkpoint directory.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor model parallel size used during conversion.")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline model parallel size used during conversion.")
    parser.add_argument("--cp-size", type=int, default=1, help="Context parallel size used during conversion.")
    parser.add_argument("--ep-size", type=int, default=1, help="Expert model parallel size used during conversion.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Model parameter dtype.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers.")
    parser.add_argument(
        "--memory-efficient-load",
        action="store_true",
        help="Load one HF tensor at a time inside mbridge.load_weights(). Slower but lower host memory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output path before conversion if it already exists and is non-empty.",
    )
    parser.add_argument(
        "--vision-attn-implementation",
        default="eager",
        help="Fallback attention implementation for Qwen3.5 vision tower when HF config requests flash_attention_2.",
    )
    parser.add_argument(
        "--make-vocab-size-divisible-by",
        type=int,
        default=None,
        help="Optional mbridge vocab padding divisor. Leave unset to use bridge default.",
    )
    return parser.parse_args()


def ensure_distributed_initialized() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29577"

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if not dist.is_initialized():
        dist.init_process_group(backend=get_nccl_backend(), rank=rank, world_size=world_size)

    device_name = get_device_name()
    torch_device = get_torch_device()
    if device_name != "cpu" and torch_device.is_available():
        torch_device.set_device(local_rank)

    return rank, world_size, local_rank


def validate_parallel_args(args: argparse.Namespace, world_size: int) -> None:
    model_parallel_size = args.tp_size * args.pp_size * args.cp_size
    if model_parallel_size <= 0:
        raise ValueError("tp_size * pp_size * cp_size must be positive.")
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must be divisible by tp_size*pp_size*cp_size={model_parallel_size}."
        )
    if args.ep_size <= 0:
        raise ValueError("ep_size must be positive.")


def prepare_output_path(path: str, overwrite: bool, rank: int) -> None:
    output = Path(path)
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Output path is not empty: {path}. Use --overwrite to replace it.")
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()


def dtype_flags(dtype: str) -> tuple[torch.dtype, bool, bool]:
    if dtype == "bf16":
        return torch.bfloat16, False, True
    if dtype == "fp16":
        return torch.float16, True, False
    return torch.float32, False, False


def patch_dist_checkpointing_results_queue() -> None:
    """Avoid extra multiprocessing workers in Megatron dist-checkpoint save.

    Megatron-Core's default torch_dist save strategy uses FileSystemWriterAsync even
    when async_sharded_save=False. In restricted container runtimes, its Manager
    queue and nested writer subprocesses can fail or hang. For this standalone
    converter, run checkpoint file writes in the current rank process instead.
    """

    from megatron.core.dist_checkpointing.strategies import filesystem_async

    queue_holder = {"items": []}

    class _InProcessResultsQueue:
        def put(self, item):
            queue_holder["items"].append(item)

        def get_nowait(self):
            if not queue_holder["items"]:
                raise filesystem_async.queue.Empty
            return queue_holder["items"].pop(0)

    def _get_write_results_queue():
        return _InProcessResultsQueue()

    def _write_preloaded_data_inprocess(transform_list, use_msc, rank, write_buckets, global_results_queue):
        write_results_or_exc = {}
        for local_proc_idx, write_bucket in enumerate(write_buckets):
            local_results_queue = _InProcessResultsQueue()
            count_queue = None
            filesystem_async.FileSystemWriterAsync.write_preloaded_data(
                transform_list,
                local_proc_idx=local_proc_idx,
                write_bucket=write_bucket,
                results_queue=local_results_queue,
                count_queue=count_queue,
                use_fsync=False,
                use_msc=use_msc,
            )
            _, local_results_or_exc = local_results_queue.get_nowait()
            if isinstance(local_results_or_exc, Exception):
                write_results_or_exc = local_results_or_exc
                break
            write_results_or_exc[local_proc_idx] = local_results_or_exc
        global_results_queue.put(write_results_or_exc)

    original_write_preloaded_data = filesystem_async.FileSystemWriterAsync.write_preloaded_data

    def _write_preloaded_data_no_count_queue(
        transform_list,
        local_proc_idx,
        write_bucket,
        results_queue,
        count_queue,
        use_fsync,
        **kwargs,
    ):
        if count_queue is not None:
            return original_write_preloaded_data(
                transform_list,
                local_proc_idx,
                write_bucket,
                results_queue,
                count_queue,
                use_fsync,
                **kwargs,
            )

        local_results = []
        try:
            file_name, storage_key, (bytes_data, tensor_data) = write_bucket
            extra_kwargs = {}
            if "serialization_format" in filesystem_async.inspect.signature(filesystem_async._write_item).parameters:
                from torch.distributed.checkpoint.filesystem import SerializationFormat

                extra_kwargs["serialization_format"] = SerializationFormat.TORCH_SAVE
            open_file = open
            if kwargs.get("use_msc", False):
                import multistorageclient as msc

                open_file = msc.open
            with open_file(file_name, "wb") as stream:
                for write_item, data in bytes_data:
                    local_results.append(
                        filesystem_async._write_item(
                            *transform_list, stream, data, write_item, storage_key, **extra_kwargs
                        )
                    )
                for write_item, tensor in tensor_data:
                    assert tensor.is_cpu
                    local_results.append(
                        filesystem_async._write_item(
                            *transform_list, stream, tensor, write_item, storage_key, **extra_kwargs
                        )
                    )
                if use_fsync:
                    if kwargs.get("use_msc", False):
                        stream.fsync()
                    else:
                        filesystem_async.os.fsync(stream.fileno())
            local_output = (local_proc_idx, local_results)
        except Exception as exc:  # pragma: no cover - propagated through checkpoint writer
            local_output = (local_proc_idx, exc)
        results_queue.put(local_output)

    filesystem_async._get_write_results_queue = _get_write_results_queue
    filesystem_async.FileSystemWriterAsync.write_preloaded_data_multiproc = staticmethod(_write_preloaded_data_inprocess)
    filesystem_async.FileSystemWriterAsync.write_preloaded_data = staticmethod(_write_preloaded_data_no_count_queue)


def save_conversion_metadata(args: argparse.Namespace, hf_config: AutoConfig, rank: int, world_size: int) -> None:
    if rank != 0:
        return
    metadata = {
        "hf_model_path": args.hf_model_path,
        "output_path": args.output_path,
        "model_type": hf_config.model_type,
        "architectures": getattr(hf_config, "architectures", None),
        "tp_size": args.tp_size,
        "pp_size": args.pp_size,
        "cp_size": args.cp_size,
        "ep_size": args.ep_size,
        "world_size": world_size,
        "dtype": args.dtype,
        "backend": get_nccl_backend(),
        "device_name": get_device_name(),
    }
    with open(Path(args.output_path) / "mbridge_conversion_args.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("QWEN3_5_VISION_ATTN_IMPLEMENTATION", args.vision_attn_implementation)

    rank, world_size, _ = ensure_distributed_initialized()
    validate_parallel_args(args, world_size)
    prepare_output_path(args.output_path, args.overwrite, rank)

    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=args.tp_size,
            pipeline_model_parallel_size=args.pp_size,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=args.cp_size,
            expert_model_parallel_size=args.ep_size,
        )
    model_parallel_cuda_manual_seed(1234)

    param_dtype, fp16, bf16 = dtype_flags(args.dtype)
    hf_config = AutoConfig.from_pretrained(args.hf_model_path, trust_remote_code=args.trust_remote_code)
    if rank == 0:
        print(f"[mbridge-convert] model_type={hf_config.model_type}, architectures={hf_config.architectures}", flush=True)
        print(
            f"[mbridge-convert] world_size={world_size}, tp={args.tp_size}, pp={args.pp_size}, "
            f"cp={args.cp_size}, ep={args.ep_size}, dtype={args.dtype}",
            flush=True,
        )

    bridge_kwargs = {"dtype": param_dtype}
    if args.make_vocab_size_divisible_by is not None:
        bridge_kwargs["make_vocab_size_divisible_by"] = args.make_vocab_size_divisible_by
    bridge = AutoBridge.from_pretrained(
        args.hf_model_path,
        trust_remote_code=args.trust_remote_code,
        **bridge_kwargs,
    )

    models = bridge.get_model(
        weight_path=None,
        wrap_with_ddp=False,
        fp16=fp16,
        bf16=bf16,
    )
    bridge.load_weights(models, args.hf_model_path, memory_efficient=args.memory_efficient_load)

    unwrapped_models = unwrap_model(models)
    if len(unwrapped_models) != 1:
        raise NotImplementedError(
            f"This converter currently expects one local model chunk, got {len(unwrapped_models)}. "
            "Run without virtual pipeline parallelism."
        )

    sharded_state_dict = unwrapped_models[0].sharded_state_dict()
    patch_dist_checkpointing_results_queue()
    dist_checkpointing.save(
        sharded_state_dict,
        args.output_path,
        sharded_strategy=None,
        async_sharded_save=False,
    )

    if rank == 0:
        hf_config.save_pretrained(args.output_path)
    save_conversion_metadata(args, hf_config, rank, world_size)
    dist.barrier()

    if rank == 0:
        print(f"[mbridge-convert] done: {args.output_path}", flush=True)

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

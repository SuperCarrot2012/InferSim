import math

from comm.comm import Comm
from flops.flops import get_attn_gflops, get_moe_gflops, get_lm_head_gflops
from hardware.gpu import gpu_map
from kvcache.kvcache import get_kvcache_size
from layers.attn import create_attention
from layers.moe import MoE
from mfu.mfu import get_gemm_mfu_and_latency, get_gemm_memory_latency
from params.params import (
    get_attn_params_size,
    get_embed_lm_head_params_size,
    get_expert_params_size,
)


class Model:
    def __init__(self, args, config):
        self.gpu = gpu_map[args.device_type]
        self.args = args
        self.config = config

    def print_weights_info(self):
        print("{s:{c}^{n}}".format(s="Model Weights", n=50, c="-"))
        tp_size = self.args.tp_size
        attn_params_bytes = get_attn_params_size(
            self.config, self.args.use_fp8_gemm, tp_size
        )
        expert_params_bytes = get_expert_params_size(
            self.config, self.args.use_fp8_gemm, tp_size
        )
        embed_bytes, lm_head_bytes, embed_lm_bytes = get_embed_lm_head_params_size(
            self.config, self.args.use_fp8_gemm, tp_size
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One attn params size (MB):", attn_params_bytes / 1024 / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One expert params size (MB):", expert_params_bytes / 1024 / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Embedding params size (MB):", embed_bytes / 1024 / 1024
            )
        )
        if self.config.tie_word_embeddings:
            print("{:<40} {:<10}".format("LM head:", "tied with embed"))
        else:
            print(
                "{:<40} {:<10.2f}".format(
                    "LM head params size (MB):", lm_head_bytes / 1024 / 1024
                )
            )
        # Unified split rule:
        #   tp_size == 1 -> attn is DP (full), MoE routed experts split by EP (= world_size)
        #   tp_size >  1 -> attn and MoE both split by TP only (no EP/world_size division)
        if tp_size == 1:
            ep_size = self.args.world_size
            experts_on_gpu = (
                self.config.num_shared_experts
                + self.config.num_routed_experts / ep_size
            )
        else:
            experts_on_gpu = (
                self.config.num_shared_experts + self.config.num_routed_experts
            )
        params_per_gpu = (
            attn_params_bytes + expert_params_bytes * experts_on_gpu
        ) * self.config.num_hidden_layers + embed_lm_bytes
        params_per_gpu = params_per_gpu / 1024 / 1024 / 1024
        self.kvcache_mem = (
            self.gpu.mem - params_per_gpu - 15 - 5
        )  # 15GB for runtime, 5GB for encoder
        print("{:<40} {:<10.2f}".format("Per GPU params size (GB):", params_per_gpu))

    def print_kvcache_info(self):
        print("{s:{c}^{n}}".format(s="KV Cache", n=50, c="-"))
        print("{:<40} {:<10.2f}".format("KV cache space (GB):", self.kvcache_mem))
        context_len = self.args.target_isl + self.args.target_osl
        target_bs = self.args.batch_size
        print("{:<40} {:<10}".format("Input seq len:", self.args.target_isl))
        print("{:<40} {:<10}".format("Output seq len:", self.args.target_osl))
        print("{:<40} {:<10}".format("Batch size per GPU:", target_bs))
        target_kvcache_bytes = (
            self.kvcache_mem * 1024 * 1024 * 1024 / target_bs / context_len
        )
        kvcache_bytes = get_kvcache_size(
            self.config, self.args.use_fp8_kv, self.args.tp_size
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Target per-token KV cache size (KB):", target_kvcache_bytes / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-token KV cache size (KB):", kvcache_bytes / 1024
            )
        )
        if kvcache_bytes > target_kvcache_bytes:
            print("!Error: need smaller kvcache")
        self.kvcache_bytes = kvcache_bytes
        self.target_bs = target_bs

    def print_flops_info(self):
        print("{s:{c}^{n}}".format(s="FLOPs", n=50, c="-"))
        print(
            "{:<40} {:<10}".format("Num hidden layers:", self.config.num_hidden_layers)
        )
        # per-token per-layer gflops
        self.avg_context_len = int(self.args.target_isl + self.args.target_osl / 2)
        attn_core_gflops, qkvo_proj_gflops = get_attn_gflops(
            self.config,
            self.target_bs,
            self.avg_context_len,
            self.args.tp_size,
            absorb=True,
        )
        moe_gflops = get_moe_gflops(self.config, self.args.tp_size)
        lm_head_gflops = get_lm_head_gflops(self.config, self.args.tp_size)
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer qkvo_proj (GFLOPs):", qkvo_proj_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer attn core (GFLOPs):", attn_core_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer MoE/FFN (GFLOPs):", moe_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token qkvo_proj (GFLOPs):",
                qkvo_proj_gflops * self.config.num_hidden_layers,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token attn core (GFLOPs):",
                attn_core_gflops * self.config.num_hidden_layers,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token MoE (GFLOPs):", moe_gflops * self.config.num_hidden_layers
            )
        )
        print("{:<40} {:<10.2f}".format("Per-token lm head (GFLOPs):", lm_head_gflops))
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token total (GFLOPs):",
                (qkvo_proj_gflops + attn_core_gflops + moe_gflops)
                * self.config.num_hidden_layers
                + lm_head_gflops,
            )
        )

    def prefill(self):
        print("{s:{c}^{n}}".format(s="Prefilling", n=50, c="-"))
        # print(
        #     "{:<40} {:<10}".format("Max prefill tokens:", self.args.max_prefill_tokens)
        # )
        attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv, self.args.tp_size
        )
        attn_qkvo_proj_time = attn.prefill_attn_qkvo_proj(
            self.target_bs, self.args.target_isl, self.args.device_type
        )
        attn_core_time = attn.prefill_attn_core(
            self.target_bs, self.args.target_isl, self.args.device_type
        )

        moe = MoE(self.config, self.args.use_fp8_gemm, self.args.tp_size)
        moe_time = moe.prefill_moe(
            self.target_bs,
            self.args.target_isl,
            self.args.device_type,
            self.args.world_size,
        )

        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )
        comm_time1, comm_time2 = comm.prefill_comm(
            self.target_bs * self.args.target_isl
        )
        print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_time1 * 1e6))
        print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_time2 * 1e6))

        # TP all_reduce communication time
        tp_comm_time = comm.tp_all_reduce(
            self.target_bs * self.args.target_isl, self.args.tp_size
        )
        if self.args.tp_size > 1:
            print("{:<40} {:<10.2f}".format("TP all_reduce (us):", tp_comm_time * 1e6))

        lm_head_latency, lm_head_mfu = get_gemm_mfu_and_latency(
            m=self.target_bs * self.args.target_isl,
            k=self.config.hidden_size,
            n=self.config.vocab_size,
            device_type=self.args.device_type,
            use_fp8_gemm=self.args.use_fp8_gemm,
        )
        lm_head_memory_latency = get_gemm_memory_latency(
            m=self.target_bs * self.args.target_isl,
            k=self.config.hidden_size,
            n=self.config.vocab_size,
            device_type=self.args.device_type,
            use_fp8_gemm=self.args.use_fp8_gemm,
        )
        lm_head_time = max(lm_head_latency, lm_head_memory_latency)
        print("{:<40} {:<10.2f}".format("LM head MFU:", lm_head_mfu))
        print("{:<40} {:<10.2f}".format("LM head latency (us):", lm_head_time * 1e6))
        print(
            "{:<40} {:<10.2f}".format(
                "LM head memory latency (us):", lm_head_memory_latency * 1e6
            )
        )

        num_tokens = self.target_bs * self.args.target_isl
        if self.args.enable_tbo:
            num_tokens *= 2
            ttft = max(
                (attn_qkvo_proj_time + attn_core_time) / self.args.sm_ratio, comm_time1
            )
            ttft += max(
                (attn_qkvo_proj_time + attn_core_time) / self.args.sm_ratio, comm_time2
            )
            ttft += max(moe_time / self.args.sm_ratio, comm_time1)
            ttft += max(moe_time / self.args.sm_ratio, comm_time2)
        else:
            ttft = attn_qkvo_proj_time + attn_core_time
            ttft += moe_time
            ttft += comm_time1 + comm_time2
            ttft += tp_comm_time  # Add TP communication time
        ttft *= self.config.num_hidden_layers
        ttft += lm_head_time
        ttft *= 1000  # convert to ms
        ttft += 30  # for scheduler

        print("{:<40} {:<10.2f}".format("TTFT (ms):", ttft))
        print(
            "{:<40} {:<10.0f}".format(
                "Throughput (TGS:tok/GPU/s):",
                num_tokens / self.args.tp_size / (ttft / 1000),
            )
        )

    def decoding(self):
        print("{s:{c}^{n}}".format(s="Decoding", n=50, c="-"))
        attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv, self.args.tp_size
        )
        attn_core_time = attn.decode_attn_core(
            self.target_bs,
            self.avg_context_len,
            self.kvcache_bytes,
            self.args.device_type,
        )
        attn_other_time = attn.decode_attn_others(self.target_bs, self.args.device_type)

        moe = MoE(self.config, self.args.use_fp8_gemm, self.args.tp_size)
        moe_time = moe.decode_moe(
            self.target_bs, self.args.device_type, self.args.world_size
        )

        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )
        comm_time1, comm_time2 = comm.decode_comm(self.target_bs)
        print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_time1 * 1e6))
        print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_time2 * 1e6))

        # TP all_reduce communication time
        tp_comm_time = comm.tp_all_reduce(self.target_bs, self.args.tp_size)
        if self.args.tp_size > 1:
            print("{:<40} {:<10.2f}".format("TP all_reduce (us):", tp_comm_time * 1e6))

        num_tokens = self.target_bs
        if self.args.enable_tbo:
            num_tokens *= 2
            tpot = max(
                attn_core_time + attn_other_time, moe_time + comm_time1 + comm_time2
            )
            tpot *= 2
        else:
            tpot = attn_core_time
            tpot += attn_other_time
            tpot += moe_time
            tpot += comm_time1 + comm_time2
            tpot += tp_comm_time  # Add TP communication time
        tpot *= self.config.num_hidden_layers
        tpot *= 1000  # convert to ms
        tpot += 5  # for scheduler

        print("{:<40} {:<10.2f}".format("TPOT (ms):", tpot))
        print(
            "{:<40} {:<10.0f}".format(
                "Throughput (TGS:tok/GPU/s):",
                num_tokens / self.args.tp_size / (tpot / 1000),
            )
        )
        if tpot > self.args.target_tpot:
            print("!Error: TPOT > SLO, need smaller GFLOPs to speedup")

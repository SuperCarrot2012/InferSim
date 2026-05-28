from flops.flops import gemm_flops, get_attn_gflops
from hardware.gpu import gpu_map
from mfu.mfu import (
    get_attn_decode_mfu,
    get_attn_prefill_mfu,
    get_gemm_mfu_and_latency,
    get_gemm_memory_latency,
    get_bmm_mfu_and_latency,
    get_bmm_memory_latency,
)


class MHA:
    def __init__(self, config, use_fp8_gemm, use_fp8_kv, tp_size):
        self.use_fp8_gemm = use_fp8_gemm
        self.use_fp8_kv = use_fp8_kv
        self.config = config
        self.tp_size = tp_size

    def attn_qkvo_proj(self, bs, seq_len, device_type):
        tp_num_heads = self.config.num_attention_heads // self.tp_size
        tp_num_kv_heads = self.config.num_key_value_heads // self.tp_size
        head_dim = self.config.head_dim
        hidden_size = self.config.hidden_size

        # q_proj shape: [bs, seq_len, hidden_size] @ [hidden_size, tp_num_heads * head_dim]
        print(
            "{:<40} {:<60}".format(
                "Q-proj shape:",
                f"[{bs},{seq_len},{hidden_size}]@[{hidden_size},{tp_num_heads * head_dim}]",
            )
        )
        q_proj_latency, q_proj_mfu = get_gemm_mfu_and_latency(
            m=bs * seq_len,
            k=hidden_size,
            n=tp_num_heads * head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.6f}".format("Q-proj MFU:", q_proj_mfu))
        print("{:<40} {:<10.2f}".format("Q-proj latency (us):", q_proj_latency * 1e6))
        q_proj_memory_latency = get_gemm_memory_latency(
            m=bs * seq_len,
            k=hidden_size,
            n=tp_num_heads * head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Q-proj memory latency (us):", q_proj_memory_latency * 1e6
            )
        )
        q_proj_latency = max(q_proj_latency, q_proj_memory_latency)

        # kv_proj shape: [bs, seq_len, hidden_size] @ [hidden_size, tp_num_kv_heads * head_dim]
        print(
            "{:<40} {:<60}".format(
                "KV-proj shape:",
                f"[{bs},{seq_len},{hidden_size}]@[{hidden_size},{tp_num_kv_heads * head_dim}]",
            )
        )
        kv_proj_latency, kv_proj_mfu = get_gemm_mfu_and_latency(
            m=bs * seq_len,
            k=hidden_size,
            n=tp_num_kv_heads * head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        kv_proj_latency *= 2
        print("{:<40} {:<10.6f}".format("KV-proj MFU:", kv_proj_mfu))
        print("{:<40} {:<10.2f}".format("KV-proj latency (us):", kv_proj_latency * 1e6))
        kv_proj_memory_latency = get_gemm_memory_latency(
            m=bs * seq_len,
            k=hidden_size,
            n=tp_num_kv_heads * head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        kv_proj_memory_latency *= 2
        print(
            "{:<40} {:<10.2f}".format(
                "KV-proj memory latency (us):", kv_proj_memory_latency * 1e6
            )
        )
        kv_proj_latency = max(kv_proj_latency, kv_proj_memory_latency)

        # o_proj shape: [bs, seq_len, tp_num_heads * head_dim] @ [tp_num_heads * head_dim, hidden_size]
        print(
            "{:<40} {:<60}".format(
                "O-proj shape:",
                f"[{bs},{seq_len},{tp_num_heads * head_dim}]@[{tp_num_heads * head_dim},{hidden_size}]",
            )
        )
        o_proj_latency, o_proj_mfu = get_gemm_mfu_and_latency(
            m=bs * seq_len,
            k=tp_num_heads * head_dim,
            n=hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.6f}".format("O-proj MFU:", o_proj_mfu))
        print("{:<40} {:<10.2f}".format("O-proj latency (us):", o_proj_latency * 1e6))
        o_proj_memory_latency = get_gemm_memory_latency(
            m=bs * seq_len,
            k=tp_num_heads * head_dim,
            n=hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print(
            "{:<40} {:<10.2f}".format(
                "O-proj memory latency (us):", o_proj_memory_latency * 1e6
            )
        )
        o_proj_latency = max(o_proj_latency, o_proj_memory_latency)
        return q_proj_latency + kv_proj_latency + o_proj_latency

    def decode_attn_core(self, bs, seq_len, device_type):
        seq_q_len = 1
        seq_kv_len = seq_len
        tp_num_heads = self.config.num_attention_heads // self.tp_size
        tp_num_kv_heads = self.config.num_key_value_heads // self.tp_size
        head_dim = self.config.head_dim
        hidden_size = self.config.hidden_size

        # Fallback to qkv gemm.
        # Q @ K^T shape: [bs, tp_num_heads, seq_q_len, head_dim] @ [bs, tp_num_heads, head_dim, seq_kv_len]
        q_kt_latency, q_kt_mfu = get_bmm_mfu_and_latency(
            b1=bs,
            b2=tp_num_heads,
            m=seq_q_len,
            k=head_dim,
            n=seq_kv_len,
            device_type=device_type,
            use_fp8_gemm=False,
        )
        print(
            "{:<40} {:<60}".format(
                "Q @ K^T shape:",
                f"[{bs},{tp_num_heads},{seq_q_len},{head_dim}]@[{bs},{tp_num_heads},{head_dim},{seq_kv_len}]",
            )
        )
        print("{:<40} {:<10.6f}".format("Q @ K^T MFU:", q_kt_mfu))
        print("{:<40} {:<10.2f}".format("Q @ K^T latency (us):", q_kt_latency * 1e6))
        q_kt_memory_latency = get_bmm_memory_latency(
            b1=bs,
            b2=tp_num_heads,
            m=seq_q_len,
            k=head_dim,
            n=seq_kv_len,
            device_type=device_type,
            use_fp8_gemm=False,
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Q @ K^T memory latency (us):", q_kt_memory_latency * 1e6
            )
        )
        q_kt_latency = max(q_kt_latency, q_kt_memory_latency)

        # P @ V shape: [bs, tp_num_heads, seq_q_len, seq_kv_len] @ [bs, tp_num_heads, seq_kv_len, head_dim]
        p_v_latency, p_v_mfu = get_bmm_mfu_and_latency(
            b1=bs,
            b2=tp_num_heads,
            m=seq_q_len,
            k=seq_kv_len,
            n=head_dim,
            device_type=device_type,
            use_fp8_gemm=False,
        )
        print(
            "{:<40} {:<60}".format(
                "P @ V shape:",
                f"[{bs},{tp_num_heads},{seq_q_len},{seq_kv_len}]@[{bs},{tp_num_heads},{seq_kv_len},{head_dim}]",
            )
        )
        print("{:<40} {:<10.6f}".format("P @ V MFU:", p_v_mfu))
        print("{:<40} {:<10.2f}".format("P @ V latency (us):", p_v_latency * 1e6))
        p_v_memory_latency = get_bmm_memory_latency(
            b1=bs,
            b2=tp_num_heads,
            m=seq_q_len,
            k=seq_kv_len,
            n=head_dim,
            device_type=device_type,
            use_fp8_gemm=False,
        )
        print(
            "{:<40} {:<10.2f}".format(
                "P @ V memory latency (us):", p_v_memory_latency * 1e6
            )
        )
        p_v_latency = max(p_v_latency, p_v_memory_latency)
        return q_kt_latency + p_v_latency

    def decode_attn_qkvo_proj(self, bs, device_type):
        return self.attn_qkvo_proj(bs, 1, device_type)

    def prefill_attn_core(self, bs, seq_len, device_type):
        tp_num_heads = self.config.num_attention_heads // self.tp_size
        tp_num_kv_heads = self.config.num_key_value_heads // self.tp_size
        head_dim = self.config.head_dim
        hidden_size = self.config.hidden_size

        # Fallback to qkv gemm.
        # Q @ K^T shape: [bs, tp_num_heads, seq_len, head_dim] @ [bs, tp_num_heads, head_dim, seq_len]
        q_kt_latency, q_kt_mfu = get_bmm_mfu_and_latency(
            b1=bs,
            b2=tp_num_heads,
            m=seq_len,
            k=head_dim,
            n=seq_len,
            device_type=device_type,
            use_fp8_gemm=False,
        )
        print(
            "{:<40} {:<60}".format(
                "Q @ K^T shape:",
                f"[{bs},{tp_num_heads},{seq_len},{head_dim}]@[{bs},{tp_num_heads},{head_dim},{seq_len}]",
            )
        )
        print("{:<40} {:<10.6f}".format("Q @ K^T MFU:", q_kt_mfu))
        print("{:<40} {:<10.2f}".format("Q @ K^T latency (us):", q_kt_latency * 1e6))
        q_kt_memory_latency = get_bmm_memory_latency(
            b1=bs,
            b2=tp_num_heads,
            m=seq_len,
            k=head_dim,
            n=seq_len,
            device_type=device_type,
            use_fp8_gemm=False,
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Q @ K^T memory latency (us):", q_kt_memory_latency * 1e6
            )
        )
        q_kt_latency = max(q_kt_latency, q_kt_memory_latency)

        # P @ V shape: [bs, tp_num_heads, seq_len, seq_len] @ [bs, tp_num_heads, seq_len, head_dim]
        p_v_latency, p_v_mfu = get_bmm_mfu_and_latency(
            b1=bs,
            b2=tp_num_heads,
            m=seq_len,
            k=seq_len,
            n=head_dim,
            device_type=device_type,
            use_fp8_gemm=False,
        )
        print(
            "{:<40} {:<60}".format(
                "P @ V shape:",
                f"[{bs},{tp_num_heads},{seq_len},{seq_len}]@[{bs},{tp_num_heads},{seq_len},{head_dim}]",
            )
        )
        print("{:<40} {:<10.6f}".format("P @ V MFU:", p_v_mfu))
        print("{:<40} {:<10.2f}".format("P @ V latency (us):", p_v_latency * 1e6))
        p_v_memory_latency = get_bmm_memory_latency(
            b1=bs,
            b2=tp_num_heads,
            m=seq_len,
            k=seq_len,
            n=head_dim,
            device_type=device_type,
            use_fp8_gemm=False,
        )
        print(
            "{:<40} {:<10.2f}".format(
                "P @ V memory latency (us):", p_v_memory_latency * 1e6
            )
        )
        p_v_latency = max(p_v_latency, p_v_memory_latency)
        return q_kt_latency + p_v_latency

    def prefill_attn_qkvo_proj(self, bs, seq_len, device_type):
        return self.attn_qkvo_proj(bs, seq_len, device_type)


class MLA(MHA):
    def __init__(self, config, use_fp8_gemm, use_fp8_kv, tp_size):
        self.use_fp8_gemm = use_fp8_gemm
        self.use_fp8_kv = use_fp8_kv
        self.config = config
        self.tp_size = tp_size

    def get_attn_core_gflops_absorb(self, bs, kv_len):
        attn_core = gemm_flops(
            bs,
            self.config.num_attention_heads
            * (self.config.kv_lora_rank + self.config.qk_rope_head_dim),
            kv_len,
        ) + gemm_flops(
            bs, kv_len, self.config.num_attention_heads * self.config.kv_lora_rank
        )
        return attn_core / 1e9

    def get_attn_core_gflops_noabsorb(self, bs, kv_len):
        attn_core = gemm_flops(
            bs,
            self.config.num_attention_heads
            * (self.config.qk_nope_head_dim + self.config.qk_rope_head_dim),
            kv_len,
        ) + gemm_flops(
            bs, kv_len, self.config.num_attention_heads * self.config.v_head_dim
        )
        return attn_core / 1e9

    def decode_attn_core(self, bs, kv_len, kvcache_bytes, device_type):
        gpu = gpu_map[device_type]
        attn_core_gflops = self.get_attn_core_gflops_absorb(1, kv_len)
        attn_core_mfu = get_attn_decode_mfu(
            self.config, bs, kv_len, device_type, self.use_fp8_kv, self.tp_size
        )
        attn_core_time = (
            bs * attn_core_gflops / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        )
        kv_load_time = (
            kvcache_bytes
            * kv_len
            * bs
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )

        print("{:<40} {:<10.6f}".format("Attn core MFU:", attn_core_mfu))
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))

        return max(attn_core_time, kv_load_time)

    def decode_attn_others(self, bs, device_type):
        # TP shards attention heads; hidden_size and lora ranks are NOT sharded
        tp_num_heads = self.config.num_attention_heads // self.tp_size
        q_down_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.hidden_size,
            n=self.config.q_lora_rank,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("Q_down_proj latency (us):", q_down_proj * 1e6))

        q_up_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.q_lora_rank,
            n=tp_num_heads * self.config.qk_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("Q_up_proj latency (us):", q_up_proj * 1e6))

        kv_down_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.hidden_size,
            n=self.config.kv_lora_rank + self.config.qk_rope_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print(
            "{:<40} {:<10.2f}".format("KV_down_proj latency (us):", kv_down_proj * 1e6)
        )

        bmm_q_wk = get_gemm_mfu_and_latency(
            m=bs,
            k=tp_num_heads * self.config.qk_nope_head_dim,
            n=self.config.kv_lora_rank,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("bmm_q_wk latency (us):", bmm_q_wk * 1e6))

        bmm_o_wv = get_gemm_mfu_and_latency(
            m=bs,
            k=tp_num_heads * self.config.kv_lora_rank,
            n=self.config.v_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("bmm_o_wv latency (us):", bmm_o_wv * 1e6))

        o_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=tp_num_heads * self.config.v_head_dim,
            n=self.config.hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("O_proj latency (us):", o_proj * 1e6))
        return q_down_proj + q_up_proj + kv_down_proj + bmm_q_wk + bmm_o_wv + o_proj

    def prefill_attn_core(self, seq_len, kvcache_bytes, device_type):
        gpu = gpu_map[device_type]
        attn_core_gflops = self.get_attn_core_gflops_noabsorb(1, seq_len)
        attn_core_mfu = get_attn_prefill_mfu(
            self.config, seq_len, device_type, self.tp_size
        )
        attn_core_time = (
            seq_len * attn_core_gflops / 1.8 / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        )
        kv_load_time = (
            kvcache_bytes
            * seq_len
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )

        print("{:<40} {:<10.6f}".format("Attn core MFU:", attn_core_mfu))
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))

        return max(attn_core_time, kv_load_time)


def create_attention(config, use_fp8_gemm, use_fp8_kv, tp_size):
    if config.attn_type == "MHA/GQA":
        return MHA(config, use_fp8_gemm, use_fp8_kv, tp_size)
    elif config.attn_type == "MLA":
        return MLA(config, use_fp8_gemm, use_fp8_kv, tp_size)

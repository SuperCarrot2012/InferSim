from flops.flops import gemm_flops
from hardware.gpu import gpu_map
from layers.attn import get_gemm_mfu_and_latency
from mfu.mfu import (
    get_gemm_mfu,
    get_gemm_memory_latency,
    get_elementwise_memory_latency,
    get_groupedgemm_decode_mfu,
    get_groupedgemm_prefill_mfu,
)
from params.params import load_moe_weights_time


class MoE:
    """
    MoE/FFN layer, dense FFN is treated as a special 1-expert MoE
    """

    def __init__(self, config, use_fp8_gemm, tp_size):
        self.use_fp8_gemm = use_fp8_gemm
        self.config = config
        self.tp_size = tp_size

    def decode_moe(self, bs, device_type, num_gpus):
        return self.prefill_moe(bs, 1, device_type, num_gpus)

    def prefill_moe(self, bs, seq_len, device_type, num_gpus):
        gpu = gpu_map[device_type]

        # TP shards intermediate_size; hidden_size is NOT sharded
        tp_intermediate_size = self.config.intermediate_size // self.tp_size
        hidden_size = self.config.hidden_size

        gate_proj_latency, gate_proj_mfu = get_gemm_mfu_and_latency(
            m=bs * seq_len,
            k=hidden_size,
            n=tp_intermediate_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        gate_proj_memory_latency = get_gemm_memory_latency(
            m=bs * seq_len,
            k=hidden_size,
            n=tp_intermediate_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print(
            "{:<40} {:<60}".format(
                "FFN Gate/Up-proj shape:",
                f"[{bs},{seq_len},{hidden_size}]@[{hidden_size},{tp_intermediate_size}]",
            )
        )
        print("{:<40} {:<10.6f}".format("FFN Gate/Up-proj MFU:", gate_proj_mfu))
        print(
            "{:<40} {:<10.2f}".format(
                "FFN Gate/Up-proj latency (us):", gate_proj_latency * 1e6
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "FFN Gate/Up-proj memory latency (us):", gate_proj_memory_latency * 1e6
            )
        )
        gate_proj_latency = max(gate_proj_latency, gate_proj_memory_latency)
        up_proj_latency = gate_proj_latency

        down_proj_latency, down_proj_mfu = get_gemm_mfu_and_latency(
            m=bs * seq_len,
            k=tp_intermediate_size,
            n=hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        down_proj_memory_latency = get_gemm_memory_latency(
            m=bs * seq_len,
            k=tp_intermediate_size,
            n=hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print(
            "{:<40} {:<60}".format(
                "FFN Down-proj shape:",
                f"[{bs},{seq_len},{tp_intermediate_size}]@[{tp_intermediate_size},{hidden_size}]",
            )
        )
        print("{:<40} {:<10.6f}".format("FFN Down-proj MFU:", down_proj_mfu))
        print(
            "{:<40} {:<10.2f}".format(
                "FFN Down-proj latency (us):", down_proj_latency * 1e6
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "FFN Down-proj memory latency (us):", down_proj_memory_latency * 1e6
            )
        )
        down_proj_latency = max(down_proj_latency, down_proj_memory_latency)

        # act_fn laytency, memory bound
        act_fn_memory_latency = get_elementwise_memory_latency(
            [
                [2, [bs, seq_len, tp_intermediate_size]],  # input
                [2, [bs, seq_len, tp_intermediate_size]],  # output
            ],
            device_type,
        )
        # mul laytency, memory bound
        mul_memory_latency = get_elementwise_memory_latency(
            [
                [2, [bs, seq_len, tp_intermediate_size]],  # input1
                [2, [bs, seq_len, tp_intermediate_size]],  # input2
                [2, [bs, seq_len, tp_intermediate_size]],  # output
            ],
            device_type,
        )
        other_latency = act_fn_memory_latency + mul_memory_latency
        print(
            "{:<40} {:<10.2f}".format(
                "FFN Act_fn&Mul latency (us):", other_latency * 1e6
            )
        )

        t = gate_proj_latency + up_proj_latency + down_proj_latency
        t += other_latency

        if self.config.num_shared_experts > 0:
            # TP shards intermediate_size; hidden_size is NOT sharded
            shared_expert_up_proj = get_gemm_mfu_and_latency(
                m=seq_len,
                k=self.config.hidden_size,
                n=tp_intermediate_size * 2 * self.config.num_shared_experts,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )

            shared_expert_down_proj = get_gemm_mfu_and_latency(
                m=seq_len,
                k=tp_intermediate_size * self.config.num_shared_experts,
                n=self.config.hidden_size,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Shared expert latency (us):",
                    (shared_expert_up_proj + shared_expert_down_proj) * 1e6,
                )
            )
            t += shared_expert_up_proj + shared_expert_down_proj
        return t

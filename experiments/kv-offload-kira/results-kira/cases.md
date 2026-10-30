| case | KIRA TTFT p50 | BLIS p50 (seed range) | error | KIRA p99 | BLIS p99 | error | KIRA decode/req | BLIS decode/req | error | external-KV tokens | n |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| E32 agent_spill, ibm1 SATA | 18.8 | 17.7 (17.7-17.7) | -6% | n/a | 80.6 | n/a | 8.83 | 13 | +47% | 92% | 220 |
| E32 agent_spill, ibm1 NVMe | 4.3 | 4.25 (4.25-4.25) | -1% | n/a | 30.9 | n/a | 8.69 | 12.9 | +49% | 95% | 220 |
| E33 chat, ibm2 SATA | 0.539 | 0.544 (0.542-0.546) | +1% | n/a | 1.18 | n/a | 9.84 | 11.5 | +17% | 79% | 1500 |
| E33 chat, ibm2 NVMe | 0.384 | 0.407 (0.405-0.409) | +6% | n/a | 0.588 | n/a | 9.66 | 11.5 | +19% | 79% | 1500 |
| E1 agent_spill, 32 GiB, stock | 18.8 | 17.7 (17.7-17.7) | -6% | 38.9 | 80.6 | +107% | n/a | 13 | n/a | 92% | 220 |
| E1 agent_spill, 36 GiB, stock | 18.8 | 17.7 (17.7-17.7) | -6% | 38.5 | 54.1 | +41% | n/a | 13 | n/a | 94% | 220 |
| E1 agent_hot, 32 GiB, stock | 11.8 | 13.8 (3.18-27.8) | +17% | 31.7 | 65.2 | +106% | n/a | 13 | n/a | 66% | 60 |
| E1 agent_hot, 36 GiB, stock | 4.9 | 20.3 (3.18-28.4) | +315% | 30.5 | 82.5 | +170% | n/a | 13.1 | n/a | 49% | 60 |
| E1 agent_spill, recompute | 27.45 | 26.8 (25.7-48.9) | -2% | 119.5 | 125 | +5% | n/a | 13.1 | n/a | 0% | 220 |
| E1 agent_hot, recompute | 74.75 | 25.7 (25.7-31.5) | -66% | 142.0 | 136 | -4% | n/a | 13.3 | n/a | 0% | 60 |
| batch 1 agent_hot, stock | 42.15 | 17.7 (17.7-22) | -58% | 64.0 | 59.3 | -7% | n/a | 13.5 | n/a | 79% | 80 |
| batch 1 agent_hot, recompute | 65.9 | 25.7 (25.7-34.9) | -61% | n/a | 116 | n/a | n/a | 13.7 | n/a | 0% | 80 |
| E1 agent_spill, 32 GiB, stock (NVMe) | n/a | 4.25 (4.25-4.25) | n/a | n/a | 30.9 | n/a | n/a | 12.9 | n/a | 95% | 220 |
| E1 agent_spill, 36 GiB, stock (NVMe) | n/a | 4.25 (4.25-4.25) | n/a | n/a | 38.4 | n/a | n/a | 12.9 | n/a | 94% | 220 |
| E1 agent_hot, 32 GiB, stock (NVMe) | n/a | 3.53 (0.798-20.3) | n/a | n/a | 56.3 | n/a | n/a | 13 | n/a | 66% | 60 |
| E1 agent_hot, 36 GiB, stock (NVMe) | n/a | 15.5 (1.83-25.8) | n/a | n/a | 79.6 | n/a | n/a | 13.1 | n/a | 50% | 60 |
| batch 1 agent_hot, stock (NVMe) | n/a | 4.25 (4.25-9.89) | n/a | n/a | 39.9 | n/a | n/a | 13.1 | n/a | 80% | 80 |

| NVMe stock, warm pass | KIRA | BLIS (seed range) | error | quantity |
|---|--:|--:|--:|---|
| E33 chat, ibm2 NVMe | 0.077 | 0.082 (0.0807-0.0844) | +7% | s, scheduled to first token (vLLM request_prefill_time) |

| case | KIRA TTFT p50 | BLIS p50 (seed range) | error | KIRA p99 | BLIS p99 | error | KIRA decode/req | BLIS decode/req | error | external-KV tokens | n |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| E32 agent_spill, ibm1 SATA | 18.8 | 18.1 (18.1-18.2) | -3% | n/a | 85.9 | n/a | 8.83 | 12.8 | +45% | 93% | 220 |
| E32 agent_spill, ibm1 NVMe | 4.3 | 4.69 (4.69-4.69) | +9% | n/a | 33.5 | n/a | 8.69 | 12.7 | +46% | 95% | 220 |
| E33 chat, ibm2 SATA | 0.539 | 0.55 (0.545-0.556) | +2% | n/a | 1.19 | n/a | 9.84 | 11.7 | +19% | 79% | 1500 |
| E33 chat, ibm2 NVMe | 0.384 | 0.397 (0.392-0.402) | +3% | n/a | 0.597 | n/a | 9.66 | 11.7 | +21% | 79% | 1500 |
| E1 agent_spill, 32 GiB, stock | 18.8 | 18.1 (18.1-18.2) | -3% | 38.9 | 85.9 | +121% | n/a | 12.8 | n/a | 93% | 220 |
| E1 agent_spill, 36 GiB, stock | 18.8 | 18.1 (18.1-18.2) | -3% | 38.5 | 57.5 | +49% | n/a | 12.8 | n/a | 94% | 220 |
| E1 agent_hot, 32 GiB, stock | 11.8 | 15.2 (3.53-30.3) | +29% | 31.7 | 71.6 | +126% | n/a | 12.8 | n/a | 66% | 60 |
| E1 agent_hot, 36 GiB, stock | 4.9 | 23 (3.53-31.2) | +369% | 30.5 | 90.7 | +197% | n/a | 12.9 | n/a | 49% | 60 |
| E1 agent_spill, recompute | 27.45 | 35.2 (28.4-57.4) | +28% | 119.5 | 147 | +23% | n/a | 13 | n/a | 0% | 220 |
| E1 agent_hot, recompute | 74.75 | 28.4 (28.4-36.8) | -62% | 142.0 | 154 | +8% | n/a | 13.2 | n/a | 0% | 60 |
| batch 1 agent_hot, stock | 42.15 | 18.1 (18.1-23.6) | -57% | 64.0 | 63.5 | -1% | n/a | 13.3 | n/a | 79% | 80 |
| batch 1 agent_hot, recompute | 65.9 | 28.5 (28.4-42.1) | -57% | n/a | 135 | n/a | n/a | 13.5 | n/a | 0% | 80 |
| E1 agent_spill, 32 GiB, stock (NVMe) | n/a | 4.69 (4.69-4.69) | n/a | n/a | 33.5 | n/a | n/a | 12.7 | n/a | 95% | 220 |
| E1 agent_spill, 36 GiB, stock (NVMe) | n/a | 4.69 (4.69-4.69) | n/a | n/a | 41.2 | n/a | n/a | 12.7 | n/a | 94% | 220 |
| E1 agent_hot, 32 GiB, stock (NVMe) | n/a | 3.88 (1.16-22.8) | n/a | n/a | 62.7 | n/a | n/a | 12.8 | n/a | 66% | 60 |
| E1 agent_hot, 36 GiB, stock (NVMe) | n/a | 17.8 (2.19-28.7) | n/a | n/a | 87.8 | n/a | n/a | 12.9 | n/a | 50% | 60 |
| batch 1 agent_hot, stock (NVMe) | n/a | 4.69 (4.69-12.3) | n/a | n/a | 42.8 | n/a | n/a | 12.9 | n/a | 80% | 80 |

| NVMe stock, warm pass | KIRA | BLIS (seed range) | error | quantity |
|---|--:|--:|--:|---|
| E33 chat, ibm2 NVMe | 0.077 | 0.0796 (0.0782-0.0815) | +3% | s, scheduled to first token (vLLM request_prefill_time) |

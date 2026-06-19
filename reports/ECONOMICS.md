# On-Prem vs API economics (break-even)

## Assumptions

| Assumption | Value |
| --- | --- |
| CAPEX (On-Prem GPU) | $1600 |
| Hardware life | 3 years |
| Electricity | $0.15 / kWh |
| GPU power draw | 70 W |
| Request | 500 in + 200 out tokens |
| Runtime basis (per request) | 10 s |
| Amortization volume | 1000 requests/day |
| Cached input fraction | 0.5 |

## Per-request cost

| Option | USD / request |
| --- | --- |
| On-Prem (amortized CAPEX + electricity) | 0.001490 |
| API `google_gemini_flash` | 0.000097 |
| API `openai_gpt4o_mini` | 0.000195 |
| API `anthropic_claude_haiku` | 0.001200 |
| Cloud-GPU (T4 rental) | 0.000972 |

## Break-even and prompt/context caching

On-Prem pays for itself after **23,414,634 requests** (~780,488/day over a month) versus the cheapest API. Prompt/context caching cuts the effective API price, **raising** the break-even to **27,137,809 requests** — a cheaper API stays competitive longer, so On-Prem must do *more* work to win.

## Recommendation (privacy vs cost)

**Recommendation.** Our measured AirLLM 4-bit throughput is only ~0.041 tok/s on a constrained 16 GB GPU, so local serving is impractical for high throughput — for cost-per-token at low volume the API wins. On-Prem's real value is **privacy**: nothing leaves the organisation, justifying the build for sensitive data regardless of the break-even. Verdict: API for throughput/low volume; On-Prem for privacy.

# Informative trajectory demonstration

The proposed method commits a 4-second informative segment, then replans with interval width 0.017076 (initial width 1).

From the identical measured state at t=4s, the prior-uncertainty backup costs 95.326 with horizon 19.243s; the updated backup costs 33.558 with horizon 6.692s. Both pass full-tube certificates and use alpha=2.

| Method | Width at learning replan | Time to width ≤ 0.05 | Arrival | Actual cost | Violations |
|---|---:|---:|---:|---:|---:|
| Robust Tube MPC | 1.000000 | not reached | 18.68s | 137.555 | 0 |
| Passive Learning | 0.051313 | 8.00s | 10.78s | 97.454 | 0 |
| Weighted Information Gain | 0.033114 | 4.00s | 13.98s | 125.539 | 0 |
| Proposed Dual-Gatekeeper | 0.017076 | 4.00s | 9.96s | 88.219 | 0 |

The excitation ablation repeats the two initial motions for equal durations using five sensor-noise seeds. It measures information content without subsequent replanning; it is not five complete mission trials.

This is a deliberately designed example with constant bounded wind, a broad drag prior, and bounded velocity-measurement noise. It demonstrates the mechanism under these assumptions, not general superiority or exact paper reproduction.

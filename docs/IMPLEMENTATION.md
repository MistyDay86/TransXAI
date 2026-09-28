# Implemented TransXAI estimand

This document records the behavior of the frozen scripts that generated the released results. It is normative for code-level reproduction.

## Decisions and representations

Explanation maps are generated for the frozen out-of-fold classifier prediction. During the refinement stage, the classifier decision used for matching and effect orientation is recomputed as `int(logit >= 0)` from the frozen model output.

For 320 x 320 inputs, recipient matching uses normalized global-average-pooled activations after the second convolutional stage (40 x 40 spatial resolution). Interventions use the output of the third convolutional stage (20 x 20, hence 400 spatial positions).

## Effects and CPTS

For a binary exact-k mask `M`, the implementation zeroes its selected spatial positions across all channels. If `z_i` is the intervention representation and `s_i` is the frozen decision sign, the signed removal effect is

```text
delta_i(M) = s_i * [g(z_i) - g((1 - M) * z_i)].
```

The same mask is evaluated independently on a compatible recipient. Pairwise preservation is

```text
P(d, r; M) = exp(-abs(delta_r(M) - delta_d(M)) /
                    (abs(delta_d(M)) + 0.05)).
```

Donor-level CPTS is the mean of `P` over the fixed recipient set. The support objective minimizes mean loss plus a CVaR tail penalty and total variation:

```text
mean(1 - P) + 0.35 * CVaR_0.80(1 - P) + 0.002 * TV(M).
```

## Feasible set and validation

- Every evaluated candidate is binary and exact-k.
- At most `max(1, round(0.25 * k))` locations may be exchanged relative to the original support.
- For base effects with magnitude at least `1e-3`, the candidate must retain the original sign and at least 90% of the original absolute donor effect.
- For smaller base effects, the candidate magnitude may not fall below the base magnitude.
- The one-sided constraint permits an increase in donor-effect magnitude.
- Search uses at most 15 iterations, swap sizes 1, 2, and 4, and three gradient offsets.
- A support step must reduce support risk by at least `1e-5`.
- The independent validation cohort must improve mean CPTS by at least `0.002`; otherwise the original support is returned.
- Query recipients are accessed only for final evaluation.

## Recipient construction

Recipients must have a different effective patient/case group and the same recomputed binary decision. Candidates are ranked by cosine similarity of the matching embedding, with at most one image per recipient group.

For outer fold `f`, query recipients come from fold `f`, validation recipients from `(f + 1) mod 5`, and support recipients from the remaining three folds. Required cohort sizes are 10, 12, and 12 respectively.

## Naming

`T-CPT` is the immutable internal identifier found in executed configs and paths. `TransXAI` is the public method name. Renaming frozen identifiers would break manifest hashes and is therefore intentionally avoided.

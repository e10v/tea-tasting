# Multiple testing

## Multiple hypothesis testing problem

/// admonition | Note

This guide uses [Polars](https://github.com/pola-rs/polars) as an example data backend. Install Polars in addition to tea-tasting to reproduce the examples:

```bash
uv pip install polars
```

///

The [multiple hypothesis testing problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem) arises when there is more than one success metric or more than one treatment variant in an A/B test.

tea-tasting provides the following methods for multiple testing correction:

- [False discovery rate](https://en.wikipedia.org/wiki/False_discovery_rate) (FDR) controlling procedures:
    - Benjamini-Hochberg procedure, assuming non-negative correlation between hypotheses.
    - Benjamini-Yekutieli procedure, assuming arbitrary dependence between hypotheses.
- [Family-wise error rate](https://en.wikipedia.org/wiki/Family-wise_error_rate) (FWER) controlling procedures:
    - Hochberg's step-up procedure, assuming non-negative correlation between hypotheses.
    - Holm's step-down procedure, assuming arbitrary dependence between hypotheses.

As an example, consider an experiment with three variants, a control and two treatments:

```pycon
>>> import polars as pl
>>> import tea_tasting as tt

>>> data = pl.concat((
...     tt.make_users_data(
...         rng=42,
...         orders_uplift=0.10,
...         revenue_uplift=0.15,
...         return_type="polars",
...     ),
...     tt.make_users_data(
...         rng=21,
...         orders_uplift=0.15,
...         revenue_uplift=0.20,
...         return_type="polars",
...     )
...         .filter(pl.col("variant").eq(1))
...         .with_columns(variant=pl.lit(2, pl.Int64)),
... ))
>>> data
shape: (6_046, 5)
┌──────┬─────────┬──────────┬────────┬─────────┐
│ user ┆ variant ┆ sessions ┆ orders ┆ revenue │
│ ---  ┆ ---     ┆ ---      ┆ ---    ┆ ---     │
│ i64  ┆ i64     ┆ i64      ┆ i64    ┆ f64     │
╞══════╪═════════╪══════════╪════════╪═════════╡
│ 0    ┆ 1       ┆ 2        ┆ 1      ┆ 9.58    │
│ 1    ┆ 0       ┆ 2        ┆ 1      ┆ 6.43    │
│ 2    ┆ 1       ┆ 2        ┆ 1      ┆ 8.3     │
│ 3    ┆ 1       ┆ 2        ┆ 1      ┆ 16.65   │
│ 4    ┆ 0       ┆ 1        ┆ 1      ┆ 7.14    │
│ …    ┆ …       ┆ …        ┆ …      ┆ …       │
│ 3989 ┆ 2       ┆ 4        ┆ 4      ┆ 34.93   │
│ 3991 ┆ 2       ┆ 1        ┆ 0      ┆ 0.0     │
│ 3992 ┆ 2       ┆ 3        ┆ 3      ┆ 27.96   │
│ 3994 ┆ 2       ┆ 2        ┆ 1      ┆ 17.22   │
│ 3998 ┆ 2       ┆ 3        ┆ 0      ┆ 0.0     │
└──────┴─────────┴──────────┴────────┴─────────┘

```

Let's calculate the experiment results:

```pycon
>>> experiment = tt.Experiment(
...     sessions_per_user=tt.Mean("sessions"),
...     orders_per_session=tt.RatioOfMeans("orders", "sessions"),
...     orders_per_user=tt.Mean("orders"),
...     revenue_per_user=tt.Mean("revenue"),
... )
>>> results = experiment.analyze(data, control=0, all_variants=True)
>>> results
variants             metric control treatment rel_effect_size rel_effect_size_ci  pvalue
  (0, 1)  sessions_per_user    2.00      1.98          -0.66%      [-3.7%, 2.5%]   0.674
  (0, 1) orders_per_session   0.266     0.289            8.8%      [-0.89%, 19%]  0.0762
  (0, 1)    orders_per_user   0.530     0.573            8.0%       [-2.0%, 19%]   0.118
  (0, 1)   revenue_per_user    5.24      5.99             14%        [2.1%, 28%]  0.0211
  (0, 2)  sessions_per_user    2.00      2.02           0.98%      [-2.1%, 4.1%]   0.532
  (0, 2) orders_per_session   0.266     0.295             11%        [1.2%, 22%]  0.0273
  (0, 2)    orders_per_user   0.530     0.594             12%        [1.7%, 23%]  0.0213
  (0, 2)   revenue_per_user    5.24      6.25             19%        [6.6%, 33%] 0.00218

```

Suppose only the two metrics `orders_per_user` and `revenue_per_user` are considered as success metrics, while the other two metrics `sessions_per_user` and `orders_per_session` are second-order diagnostic metrics.

```pycon
>>> metrics = {"orders_per_user", "revenue_per_user"}

```

With two treatment variants and two success metrics, there are four hypotheses in total, which increases the probability of false positives (also called "false discoveries"). It is recommended to adjust the p-values or the significance level (alpha) in this case. Let's explore the correction methods provided by tea-tasting.

## False discovery rate

False discovery rate (FDR) is the expected value of the proportion of false discoveries among the discoveries (rejections of the null hypothesis). To control for FDR, use the [`adjust_fdr`](api/multiplicity.md#tea_tasting.multiplicity.adjust_fdr) method:

```pycon
>>> adjusted_results_fdr = tt.adjust_fdr(results, metrics)
>>> adjusted_results_fdr
comparison           metric control treatment rel_effect_size  pvalue pvalue_adj
    (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118      0.118
    (0, 1) revenue_per_user    5.24      5.99             14%  0.0211     0.0284
    (0, 2)  orders_per_user   0.530     0.594             12%  0.0213     0.0284
    (0, 2) revenue_per_user    5.24      6.25             19% 0.00218    0.00872

```

The method adjusts p-values and saves them as `pvalue_adj`. Compare these values to the desired significance level alpha to determine if the null hypotheses can be rejected.

The method also adjusts the significance level alpha and saves it as `alpha_adj`. Compare non-adjusted p-values (`pvalue`) to the `alpha_adj` to determine if the null hypotheses can be rejected:

```pycon
>>> adjusted_results_fdr.with_keys((
...     "comparison",
...     "metric",
...     "control",
...     "treatment",
...     "rel_effect_size",
...     "pvalue",
...     "alpha_adj",
... ))
comparison           metric control treatment rel_effect_size  pvalue alpha_adj
    (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118    0.0500
    (0, 1) revenue_per_user    5.24      5.99             14%  0.0211    0.0375
    (0, 2)  orders_per_user   0.530     0.594             12%  0.0213    0.0375
    (0, 2) revenue_per_user    5.24      6.25             19% 0.00218    0.0375

```

By default, tea-tasting assumes non-negative correlation between hypotheses and performs the Benjamini-Hochberg procedure. To perform the Benjamini-Yekutieli procedure, assuming arbitrary dependence between hypotheses, set the `arbitrary_dependence` parameter to `True`:

```pycon
>>> tt.adjust_fdr(results, metrics, arbitrary_dependence=True)
comparison           metric control treatment rel_effect_size  pvalue pvalue_adj
    (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118      0.245
    (0, 1) revenue_per_user    5.24      5.99             14%  0.0211     0.0592
    (0, 2)  orders_per_user   0.530     0.594             12%  0.0213     0.0592
    (0, 2) revenue_per_user    5.24      6.25             19% 0.00218     0.0182

```

## Family-wise error rate

Family-wise error rate (FWER) is the probability of making at least one type I error. To control for FWER, use the [`adjust_fwer`](api/multiplicity.md#tea_tasting.multiplicity.adjust_fwer) method:

```pycon
>>> tt.adjust_fwer(results, metrics)
comparison           metric control treatment rel_effect_size  pvalue pvalue_adj
    (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118      0.118
    (0, 1) revenue_per_user    5.24      5.99             14%  0.0211     0.0422
    (0, 2)  orders_per_user   0.530     0.594             12%  0.0213     0.0422
    (0, 2) revenue_per_user    5.24      6.25             19% 0.00218    0.00869

```

By default, tea-tasting assumes non-negative correlation between hypotheses and performs the Hochberg's step-up procedure with the Šidák correction, which is slightly more powerful than the Bonferroni correction.

To perform Holm's step-down procedure, assuming arbitrary dependence between hypotheses, set the `arbitrary_dependence` parameter to `True`. In this case, it is recommended to use the Bonferroni correction, since the Šidák correction assumes non-negative correlation between hypotheses:

```pycon
>>> tt.adjust_fwer(
...     results,
...     metrics,
...     arbitrary_dependence=True,
...     method="bonferroni",
... )
comparison           metric control treatment rel_effect_size  pvalue pvalue_adj
    (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118      0.118
    (0, 1) revenue_per_user    5.24      5.99             14%  0.0211     0.0634
    (0, 2)  orders_per_user   0.530     0.594             12%  0.0213     0.0634
    (0, 2) revenue_per_user    5.24      6.25             19% 0.00218    0.00872

```

## Other inputs

In the examples above, the methods `adjust_fdr` and `adjust_fwer` received results from a *single experiment* with *more than two variants*. They can also accept the results from *multiple experiments* with *two variants* in each:

```pycon
>>> data1 = tt.make_users_data(rng=42, orders_uplift=0.10, revenue_uplift=0.15)
>>> data2 = tt.make_users_data(rng=21, orders_uplift=0.15, revenue_uplift=0.20)
>>> result1 = experiment.analyze(data1)
>>> result2 = experiment.analyze(data2)
>>> tt.adjust_fdr(
...     {"Experiment 1": result1, "Experiment 2": result2},
...     metrics,
... )
  comparison           metric control treatment rel_effect_size   pvalue pvalue_adj
Experiment 1  orders_per_user   0.530     0.573            8.0%    0.118      0.118
Experiment 1 revenue_per_user    5.24      5.99             14%   0.0211     0.0282
Experiment 2  orders_per_user   0.514     0.594             16%  0.00427    0.00853
Experiment 2 revenue_per_user    5.10      6.25             22% 6.27e-04    0.00251

```

The methods `adjust_fdr` and `adjust_fwer` can also accept the result of *a single experiment with two variants*:

```pycon
>>> tt.adjust_fwer(result2, metrics)
comparison           metric control treatment rel_effect_size   pvalue pvalue_adj
         -  orders_per_user   0.514     0.594             16%  0.00427    0.00427
         - revenue_per_user    5.10      6.25             22% 6.27e-04    0.00125

```

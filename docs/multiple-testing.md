# Multiple testing

## Multiple hypothesis testing problem

???+ note

    This guide uses [Polars](https://github.com/pola-rs/polars) as an example data backend. To be able to reproduce the example code, install Polars in addition to **tea-tasting**:

    ```bash
    pip install polars
    ```

The [multiple hypothesis testing problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem) arises when there is more than one success metric or more than one treatment variant in an A/B test.

**tea-tasting** provides the following methods for multiple testing correction:

- [False discovery rate](https://en.wikipedia.org/wiki/False_discovery_rate) (FDR) controlling procedures:
    - Benjamini-Yekutieli procedure, assuming arbitrary dependence between hypotheses.
    - Benjamini-Hochberg procedure, assuming non-negative correlation between hypotheses.
- [Family-wise error rate](https://en.wikipedia.org/wiki/Family-wise_error_rate) (FWER) controlling procedures:
    - Holm's step-down procedure, assuming arbitrary dependence between hypotheses.
    - Hochberg's step-up procedure, assuming non-negative correlation between hypotheses.

As an example, consider an experiment with three variants, a control and two treatments:

```pycon
>>> import polars as pl
>>> import tea_tasting as tt

>>> data = pl.concat((
...     tt.make_users_data(
...         seed=42,
...         orders_uplift=0.10,
...         revenue_uplift=0.15,
...         return_type="polars",
...     ),
...     tt.make_users_data(
...         seed=21,
...         orders_uplift=0.15,
...         revenue_uplift=0.20,
...         return_type="polars",
...     )
...         .filter(pl.col("variant").eq(1))
...         .with_columns(variant=pl.lit(2, pl.Int64)),
... ))
>>> print(data)
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
>>> print(results)
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

With two treatment variants and two success metrics, there are four hypotheses in total, which increases the probability of false positives (also called "false discoveries"). It's recommended to adjust the p-values or the significance level (alpha) in this case. Let's explore the correction methods provided by **tea-tasting**.

## False discovery rate

False discovery rate (FDR) is the expected value of the proportion of false discoveries among the discoveries (rejections of the null hypothesis). To control for FDR, use the [`adjust_fdr`](api/multiplicity.md#tea_tasting.multiplicity.adjust_fdr) method:

```pycon
>>> adjusted_results_fdr = tt.adjust_fdr(results, metrics)
>>> print(adjusted_results_fdr)
comparison           metric control treatment rel_effect_size  pvalue pvalue_adj
    (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118      0.245
    (0, 1) revenue_per_user    5.24      5.99             14%  0.0211     0.0592
    (0, 2)  orders_per_user   0.530     0.594             12%  0.0213     0.0592
    (0, 2) revenue_per_user    5.24      6.25             19% 0.00218     0.0182

```

The method adjusts p-values and saves them as `pvalue_adj`. Compare these values to the desired significance level alpha to determine if the null hypotheses can be rejected.

The method also adjusts the significance level alpha and saves it as `alpha_adj`. Compare non-adjusted p-values (`pvalue`) to the `alpha_adj` to determine if the null hypotheses can be rejected:

```pycon
>>> print(adjusted_results_fdr.to_string(keys=(
...     "comparison",
...     "metric",
...     "control",
...     "treatment",
...     "rel_effect_size",
...     "pvalue",
...     "alpha_adj",
... )))
comparison           metric control treatment rel_effect_size  pvalue alpha_adj
    (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118    0.0240
    (0, 1) revenue_per_user    5.24      5.99             14%  0.0211    0.0120
    (0, 2)  orders_per_user   0.530     0.594             12%  0.0213    0.0180
    (0, 2) revenue_per_user    5.24      6.25             19% 0.00218   0.00600

```

By default, **tea-tasting** assumes arbitrary dependence between hypotheses and performs the Benjamini-Yekutieli procedure. To perform the Benjamini-Hochberg procedure, assuming non-negative correlation between hypotheses, set the `arbitrary_dependence` parameter to `False`:

```pycon
>>> print(tt.adjust_fdr(results, metrics, arbitrary_dependence=False))
comparison           metric control treatment rel_effect_size  pvalue pvalue_adj
    (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118      0.118
    (0, 1) revenue_per_user    5.24      5.99             14%  0.0211     0.0284
    (0, 2)  orders_per_user   0.530     0.594             12%  0.0213     0.0284
    (0, 2) revenue_per_user    5.24      6.25             19% 0.00218    0.00872

```

## Family-wise error rate

Family-wise error rate (FWER) is the probability of making at least one type I error. To control for FWER, use the [`adjust_fwer`](api/multiplicity.md#tea_tasting.multiplicity.adjust_fwer) method:

```pycon
>>> print(tt.adjust_fwer(results, metrics))
comparison           metric control treatment rel_effect_size  pvalue pvalue_adj
    (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118      0.118
    (0, 1) revenue_per_user    5.24      5.99             14%  0.0211     0.0634
    (0, 2)  orders_per_user   0.530     0.594             12%  0.0213     0.0634
    (0, 2) revenue_per_user    5.24      6.25             19% 0.00218    0.00872

```

By default, **tea-tasting** assumes arbitrary dependence between hypotheses and performs the Holm's step-down procedure with Bonferroni correction. To perform the Hochberg's step-up procedure, assuming non-negative correlation between hypotheses, set the `arbitrary_dependence` parameter to `False`. In this case, you can also use the slightly more powerful Šidák correction instead of the Bonferroni correction:

```pycon
>>> print(tt.adjust_fwer(
...     results,
...     metrics,
...     arbitrary_dependence=False,
...     method="sidak",
... ))
comparison           metric control treatment rel_effect_size  pvalue pvalue_adj
    (0, 1)  orders_per_user   0.530     0.573            8.0%   0.118      0.118
    (0, 1) revenue_per_user    5.24      5.99             14%  0.0211     0.0422
    (0, 2)  orders_per_user   0.530     0.594             12%  0.0213     0.0422
    (0, 2) revenue_per_user    5.24      6.25             19% 0.00218    0.00869

```

## Other inputs

In the examples above, the methods `adjust_fdr` and `adjust_fwer` received results from a *single experiment* with *more than two variants*. They can also accept the results from *multiple experiments* with *two variants* in each:

```pycon
>>> data1 = tt.make_users_data(seed=42, orders_uplift=0.10, revenue_uplift=0.15)
>>> data2 = tt.make_users_data(seed=21, orders_uplift=0.15, revenue_uplift=0.20)
>>> result1 = experiment.analyze(data1)
>>> result2 = experiment.analyze(data2)
>>> print(tt.adjust_fdr(
...     {"Experiment 1": result1, "Experiment 2": result2},
...     metrics,
... ))
  comparison           metric control treatment rel_effect_size   pvalue pvalue_adj
Experiment 1  orders_per_user   0.530     0.573            8.0%    0.118      0.245
Experiment 1 revenue_per_user    5.24      5.99             14%   0.0211     0.0587
Experiment 2  orders_per_user   0.514     0.594             16%  0.00427     0.0178
Experiment 2 revenue_per_user    5.10      6.25             22% 6.27e-04    0.00522

```

The methods `adjust_fdr` and `adjust_fwer` can also accept the result of *a single experiment with two variants*:

```pycon
>>> print(tt.adjust_fwer(result2, metrics))
comparison           metric control treatment rel_effect_size   pvalue pvalue_adj
         -  orders_per_user   0.514     0.594             16%  0.00427    0.00427
         - revenue_per_user    5.10      6.25             22% 6.27e-04    0.00125

```

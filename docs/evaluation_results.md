# Evaluation results (Phase 5)

Held-out test scored **once** with the Phase 4 winner (**Blend: Ridge + GradientBoosting**).

| Split | n | RMSE (log) | MAE ($) | R2 |
|---|---|---|---|---|
| cv_train_5fold | 1166 | 0.1067 | 13,067 | 0.929 |
| test | 292 | 0.1236 | 13,798 | 0.930 |
| train_resubstitution | 1166 | 0.0547 | 7,077 | 0.981 |

Test RMSE(log) = **0.1236** (+15.9% vs CV). `train_resubstitution` refits nothing: it is the same frozen model scored on the rows it was trained on, so it is optimistic (especially for the tree member).

Top raw features by permutation importance (test): OverallQual (0.0502), TotalBsmtSF (0.0349), GrLivArea (0.0314), 2ndFlrSF (0.0268), 1stFlrSF (0.0221), YearBuilt (0.0142), OverallCond (0.0133), LotArea (0.0114).

## Linear-assumption checks (Ridge member, training rows)

- Columns with VIF > 10: **162**. Expected with one-hot neighborhoods; Ridge shrinks the overlapping coefficients rather than dropping them.
- Training rows with Cook's distance > 4/n: **42**. These are influential, not automatically errors; they were not dropped.
- Test log-residuals: mean -0.0010, std 0.1238. Shapiro p=3.77e-16; Spearman |residual| vs predicted ρ=-0.029 (p=0.621).

## Houses the model gets most wrong (test)

```
       actual  predicted  pct_error SaleCondition Neighborhood                                                                          why
197   40000.0    93383.8     -133.5        Normal       IDOTRR                                              low quality (4); over-predicted
134   35311.0    59278.8      -67.9       Abnorml       IDOTRR                      unusual sale (Abnorml); low quality (2); over-predicted
181   55993.0    93593.3      -67.2        Alloca       IDOTRR                                        unusual sale (Alloca); over-predicted
278  143000.0   237618.4      -66.2       Partial      ClearCr                                      new-home / partial sale; over-predicted
227   64500.0   102104.4      -58.3        Normal      OldTown                                              low quality (4); over-predicted
223   66500.0    95274.1      -43.3       Abnorml      OldTown                                       unusual sale (Abnorml); over-predicted
260  395000.0   281567.4       28.7       Partial      NridgHt                      new-home / partial sale; sale type New; under-predicted
109  135000.0   187551.5      -38.9        Family        NAmes                                        unusual sale (Family); over-predicted
43   253293.0   339705.4      -34.1       Partial      NridgHt                       new-home / partial sale; sale type New; over-predicted
226   68400.0    90426.4      -32.2       Abnorml       IDOTRR       unusual sale (Abnorml); sale type COD; low quality (4); over-predicted
27   141000.0   107061.8       24.1        Normal      Gilbert                                             low quality (4); under-predicted
6    311500.0   237176.1       23.9        Normal      Crawfor                                                              under-predicted
127   79500.0   102862.4      -29.4        Normal      BrkSide                                                               over-predicted
58   107000.0   138199.3      -29.2       Abnorml      Edwards                                       unusual sale (Abnorml); over-predicted
139  755000.0   587387.6       22.2        Normal      NoRidge  very high quality (10); very large living area (4316 sqft); under-predicted
```

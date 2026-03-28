# Open Questions

## Phase 1

1. **Optimal control gain k**: The paper derives an optimal control law but the exact relationship between k and the OU parameters (κ, σ) needs further investigation. In Phase 6, k should be optimized as a function of estimated parameters.

2. **Parameter estimation bias**: The OLS estimator for κ is known to have a downward bias for small samples. For the rolling-window estimation in Phase 3, the lookback window size will affect estimation quality.

3. **Spread definition generality**: Phase 1 uses a simple linear spread. The paper's framework allows arbitrary (potentially non-linear) spread functions, which will be explored in Phase 8.

4. **Portfolio value model simplicity**: The current backtest uses additive PnL (h * ds) rather than multiplicative returns. For large allocations, this could diverge from realistic portfolio dynamics. This should be addressed when moving to real data.

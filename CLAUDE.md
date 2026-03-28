# A General Framework for Pairs Trading with a Control-Theoretic Point of View

## Project ID
proj_76e30a1c

## Taxonomy
StatArb

## Current Cycle
1

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
The paper proposes a generalized framework for pairs trading, viewing it through the lens of stochastic control theory. Traditional methods often rely on fixed distance-based rules for entering and exiting trades. This paper's approach is more dynamic and flexible. It models the pair's spread as a mean-reverting stochastic process (like an Ornstein-Uhlenbeck process) and uses a feedback control mechanism to continuously adjust the investment level. The framework is general, allowing for arbitrary, potentially non-linear, spread definitions. The central theoretical contribution is the proof that if the chosen spread is indeed mean-reverting, the strategy guarantees a positive expected logarithmic growth of the portfolio.

### Datasets
yfinance: Daily OHLCV data for US-listed stocks and ETFs. e.g., EWA/EWC, GLD/GDX, PEP/KO.
- FRED: VIX index for market volatility regime analysis.

### Targets
The primary target is not a prediction, but an optimization of the investment allocation `h(t)` in the spread portfolio. The objective is to maximize the long-term expected portfolio growth rate `g`.

### Model
The model is a stochastic control system. The 'state' of the system is the spread `s(t)` between two assets, which is modeled as a mean-reverting Ornstein-Uhlenbeck (OU) process. The 'control' is the investment level `h(t)`. The strategy employs a feedback control law, `h(t) = f(s(t))`, which dictates the investment size based on the current deviation of the spread from its long-term mean. The core of the implementation involves: 1) Estimating the parameters of the OU process (mean-reversion speed `κ`, mean `μ`, volatility `σ`) on a rolling basis. 2) Applying the derived optimal control law to determine the daily portfolio allocation.

### Training
The 'training' consists of estimating the parameters of the spread's stochastic process. This is performed using a walk-forward methodology. For each time step, parameters (`κ`, `μ`, `σ`) are estimated using a lookback window of historical spread data (e.g., 252 days). These estimated parameters are then used to determine the trading decisions for the subsequent period (e.g., 1 day). There is no traditional train/validation/test split for a single model; the entire process is a sequential backtest.

### Evaluation
The primary evaluation method is a walk-forward backtest over the entire available data period. Key performance metrics include: Annualized Sharpe Ratio, Calmar Ratio, Maximum Drawdown (MDD), and the average portfolio growth rate. Performance will be evaluated both gross and net of transaction costs. The robustness of the strategy will be assessed by testing it on multiple asset pairs and analyzing its performance across different market volatility regimes.


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## ★ 今回のタスク (Cycle 1)


### Phase 1: コアアルゴリズムと合成データでの検証 [Track ]

**Track**:  (A=論文再現 / B=近傍改善 / C=独自探索)
**ゴール**: スプレッドのOU過程と制御則に基づく取引ロジックを実装し、合成データで動作確認する。

**具体的な作業指示**:
1. `src/` ディレクトリを作成。 2. `src/ou_process.py` に、指定されたパラメータ（κ, μ, σ）でOrnstein-Uhlenbeck過程の時系列を生成する関数 `generate_ou_path` を実装。 3. `src/model.py` に `ControlTrader` クラスを作成。コンストラクタはOUパラメータと制御ゲイン `k` を受け取る。`get_allocation(spread_value)` メソッドを実装し、制御則 `h = -k * (spread_value - mu)` に基づいてアロケーションを返す。 4. `notebooks/01_synthetic_data_validation.ipynb` を作成。`generate_ou_path` で合成スプレッドデータを生成し、`ControlTrader` を使って簡単なバックテストを実行し、ポートフォリオ価値の推移をプロットする。 5. `README.md` に、本プロジェクトが論文の縮小版検証であることを明記する。

**期待される出力ファイル**:
- src/model.py
- src/ou_process.py
- notebooks/01_synthetic_data_validation.ipynb

**受入基準 (これを全て満たすまで完了としない)**:
- `ControlTrader`クラスが実装されている
- 合成データでのバックテストが実行され、結果がノートブックに可視化されている
- ユニットテストが`ControlTrader`の基本的な計算をカバーしている




## データ問題でスタックした場合の脱出ルール

レビューで3サイクル連続「データ関連の問題」が指摘されている場合:
1. **データの完全性を追求しすぎない** — 利用可能なデータでモデル実装に進む
2. **合成データでのプロトタイプを許可** — 実データが不足する部分は合成データで代替し、モデルの基本動作を確認
3. **データの制約を open_questions.md に記録して先に進む**
4. 目標は「論文の手法が動くこと」であり、「論文と同じデータを揃えること」ではない







## 全体Phase計画 (参考)

→ Phase 1: コアアルゴリズムと合成データでの検証 — スプレッドのOU過程と制御則に基づく取引ロジックを実装し、合成データで動作確認する。
  Phase 2: 実データパイプラインの構築 — yfinanceからペア（EWA/EWC）の株価データを取得し、前処理を行うパイプラインを実装する。
  Phase 3: OUパラメータのローリング推定 — 実データスプレッドに対して、OU過程のパラメータをローリングウィンドウで推定する機能を実装する。
  Phase 4: ウォークフォワード評価フレームワーク — 厳密なウォークフォワード検証を実装し、主要なパフォーマンス指標を計算する。
  Phase 5: 取引コストモデルの導入 — バックテストエンジンに取引コストモデルを組み込み、グロスとネットのパフォーマンスを比較する。
  Phase 6: ハイパーパラメータ最適化 — OUパラメータ推定のルックバック期間と制御ゲインを最適化する。
  Phase 7: ロバスト性検証：複数ペアでのテスト — 最適化されたパラメータを用いて、戦略を複数の異なる資産ペアで実行し、ロバスト性を評価する。
  Phase 8: 代替スプレッド関数の実装と評価 — 論文の「一般化フレームワーク」の主張を検証するため、非線形スプレッド関数を実装し、線形スプレッドと比較する。
  Phase 9: 市場レジーム分析 — 戦略のパフォーマンスを高ボラティリティ市場と低ボラティリティ市場で比較分析する。
  Phase 10: 最終レポート生成と結果の統合 — 全フェーズの結果を統合し、包括的なテクニカルレポートを生成する。
  Phase 11: エグゼクティブサマリーとコード品質向上 — 非技術者向けの要約を作成し、プロジェクトのコード品質を最終化する。


## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 再現モードのルール（論文忠実度の維持）

このプロジェクトは**論文再現**が目的。パフォーマンス改善より論文忠実度を優先すること。

### パラメータ探索の制約
- **論文で既定されたパラメータをまず実装し、そのまま評価すること**
- パラメータ最適化を行う場合、**論文既定パラメータの近傍のみ**を探索（例: 論文が12ヶ月なら [6, 9, 12, 15, 18] ヶ月）
- 論文と大きく異なるパラメータ（例: 月次論文に対して日次10営業日）で良い結果が出ても、それは「論文再現」ではなく「独自探索」
- 独自探索で得た結果は `customMetrics` に `label: "implementation-improvement"` として記録し、論文再現結果と明確に分離

### データ条件の忠実度
- 論文のデータ頻度（日次/月次/tick）にできるだけ合わせる
- ユニバース規模が論文より大幅に小さい場合、その制約を `docs/open_questions.md` に明記
- リバランス頻度・加重方法も論文に合わせる



## 禁止事項
- 未来情報を特徴量やシグナルに使わない
- 全サンプル統計でスケーリングしない (train-onlyで)
- テストセットでハイパーパラメータを調整しない
- コストなしのgross PnLだけで判断しない
- 時系列データにランダムなtrain/test splitを使わない
- APIキーやクレデンシャルをコミットしない
- **新しい `scripts/run_cycle_N.py` や `scripts/experiment_cycleN.py` を作成しない。既存の `src/` 内ファイルを修正・拡張すること**
- **合成データを自作しない。必ずARF Data APIからデータを取得すること**
- **「★ 今回のタスク」以外のPhaseの作業をしない。1サイクル=1Phase**
- **論文が既定するパラメータから大幅に逸脱した探索を「再現」として報告しない**

## Git / ファイル管理ルール
- **データファイル(.csv, .parquet, .h5, .pkl, .npy)は絶対にgit addしない**
- `__pycache__/`, `.pytest_cache/`, `*.pyc` がリポジトリに入っていたら `git rm --cached` で削除
- `git add -A` や `git add .` は使わない。追加するファイルを明示的に指定する
- `.gitignore` を変更しない（スキャフォールドで設定済み）
- データは `data/` ディレクトリに置く（.gitignore済み）
- 学習済みモデルは `models/` ディレクトリに置く（.gitignore済み）

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_1/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_1/technical_findings.md` — 実装内容、結果、観察事項

### metrics.json 必須スキーマ
```json
{
  "sharpeRatio": 0.0,
  "annualReturn": 0.0,
  "maxDrawdown": 0.0,
  "hitRate": 0.0,
  "totalTrades": 0,
  "transactionCosts": { "feeBps": 10, "slippageBps": 5, "netSharpe": 0.0 },
  "walkForward": { "windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0 },
  "customMetrics": {}
}
```
- 全フィールドを埋めること。Phase 1-2で未実装のメトリクスは0.0/0で可。
- `customMetrics`に論文固有の追加メトリクスを自由に追加してよい。
- `docs/open_questions.md` — 未解決の疑問と仮定
- `README.md` — 今回のサイクルで変わった内容を反映して更新（セットアップ手順、主要な結果、使い方など）
- `docs/open_questions.md` に以下も記録:
  - ARF Data APIで問題が発生した場合（エラー、データ不足、期間の短さ等）
  - CLAUDE.mdの指示で不明確な点や矛盾がある場合
  - 環境やツールの制約で作業が完了できなかった場合

## 標準バックテストフレームワーク

`src/backtest.py` に以下が提供済み。ゼロから書かず、これを活用すること:
- `WalkForwardValidator` — Walk-forward OOS検証のtrain/test split生成
- `calculate_costs()` — ポジション変更に基づく取引コスト計算
- `compute_metrics()` — Sharpe, 年率リターン, MaxDD, Hit rate算出
- `generate_metrics_json()` — ARF標準のmetrics.json生成

```python
from src.backtest import WalkForwardValidator, BacktestConfig, calculate_costs, compute_metrics, generate_metrics_json
```

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.

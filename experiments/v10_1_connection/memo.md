# v10.1.0 — Anti-Hub + Top-K Random

Purpose:
Reduce hub dominance and introduce stochastic selection.

Setup:
Nodes=10, Steps=400, TOP_K=6

Result:
Connections distributed more evenly.
No dominant hub observed.
Trust remains stable without saturation.
Variance ~0.108.

Conclusion:
Anti-hub and Top-K random improve structural balance.

--- JP ---
ハブ集中を抑制
Top-Kランダムで偏り軽減
信頼スコアは安定
分散も維持（約0.108）
構造バランス改善

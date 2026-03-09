# autoresearch-gnn program.md (v0.1)

目标：在 **Graph-WaveNet + METR-LA** 上执行可复现的自动实验循环，稳定降低 `Valid Loss (MAE)`。

## 0) 研究约束（必须遵守）

1. **主指标**：`Valid Loss (MAE)`（越低越好）。
2. **接受阈值**：相对当前 best 至少提升 `0.5%` 才接受。
3. **预算固定**：每次 trial 固定相同 `epochs`（默认 1，可调）。
4. **数据固定**：仅使用 `data/METR-LA`（train/val/test）。
5. **可复现**：每次实验写入 `experiments/ledger.jsonl`，并保存完整日志。
6. **安全边界**：禁止删除数据、禁止改动数据路径，禁止修改非白名单文件。

## 1) 白名单（v0.1）

v0.1 先做“低风险自动实验”，仅通过训练参数搜索，不自动改代码：
- `--nhid`
- `--dropout`
- `--learning_rate`
- `--weight_decay`
- `--batch_size`
- `--gcn_bool`
- `--addaptadj`
- `--randomadj`
- `--adjtype`

> 说明：后续 v0.2 再开启对 `model.py` 的受控改动。

## 2) 决策规则

- trial 失败（超时/报错/缺指标） => `reject`
- trial 成功但 `valid_loss` 未超过阈值 => `reject`
- trial 成功且 `valid_loss` 提升超过阈值 => `accept + 更新 best`

## 3) 实验账本格式（JSONL）

每行一个 JSON 对象，包含：
- `trial_id`
- `timestamp`
- `config`
- `duration_sec`
- `status` (`ok`/`error`/`timeout`)
- `metrics` (`valid_loss`, `test_mae`, `test_mape`, `test_rmse`)
- `decision` (`accept`/`reject`)
- `reason`
- `log_path`

## 4) 推荐运行方式

```bash
source .venv/bin/activate
python orchestrator.py --trials 10 --epochs 1 --device cpu
```

如要提速，可在有 GPU 时改为 `--device cuda:0`。

## 5) 成功定义（v0.1）

- 能连续自动跑完 N 个 trial
- 账本与日志完整
- 至少产出一个比 baseline 更优的配置（或明确得出“当前搜索空间无改进”）

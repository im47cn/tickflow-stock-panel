"""优化器编排测试 — 用假 service 注入受控 stats, 验证排序/取消/进度/目标方向。"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date

import pytest

from app.backtest.optimizer import OptimizeConfig, StrategyOptimizer

# ---- 假 StrategyDef / 引擎 / service ----

@dataclass
class _FakeDef:
    meta: dict


class _FakeEngine:
    def __init__(self, params_meta):
        self._def = _FakeDef(meta={"params": params_meta})

    def get(self, strategy_id):
        return self._def


@dataclass
class _FakeResult:
    stats: dict
    error: str | None = None


class _FakeService:
    """run() 依据 params 返回受控 stats: sortino = ma_proximity 的映射, 便于校验排序。"""

    def __init__(self, score_fn):
        self.score_fn = score_fn
        self.calls = []
        self._lock = threading.Lock()

    def run(self, config, progress_cb=None, cancel_event=None):
        with self._lock:
            self.calls.append(dict(config.params or {}))
        return self.score_fn(config.params or {})


PARAMS_META = [
    {"id": "ma_proximity", "type": "float", "default": 0.02, "min": 0.01, "max": 0.05, "step": 0.005},
]


def _optimizer(score_fn):
    return StrategyOptimizer(_FakeService(score_fn), _FakeEngine(PARAMS_META))


def _cfg(**kw):
    base = dict(
        strategy_id="s", symbols=None, start=date(2024, 1, 1), end=date(2024, 6, 1),
        param_grid={"ma_proximity": [0.01, 0.02, 0.03]}, objective="sortino", max_workers=4,
    )
    base.update(kw)
    return OptimizeConfig(**base)


def test_ranks_best_by_objective_max():
    # sortino 随 ma_proximity 递增 -> 最大值应为 0.03
    def score(p):
        return _FakeResult(stats={"sortino": p["ma_proximity"] * 100})
    out = _optimizer(score).optimize(_cfg())
    assert out["best_params"] == {"ma_proximity": 0.03}
    assert out["best_score"] == 3.0
    assert out["n_combinations"] == 3
    assert out["n_completed"] == 3
    assert [r["rank"] for r in out["results"]] == [1, 2, 3]
    assert out["results"][0]["params"] == {"ma_proximity": 0.03}


def test_all_combos_executed_once():
    def score(p):
        return _FakeResult(stats={"sortino": 1.0})
    opt = _optimizer(score)
    out = opt.optimize(_cfg(param_grid={"ma_proximity": [0.01, 0.02, 0.03, 0.04, 0.05]}))
    assert out["n_combinations"] == 5
    # 每组恰跑一次
    ran = sorted(c["ma_proximity"] for c in opt.service.calls)
    assert ran == [0.01, 0.02, 0.03, 0.04, 0.05]


def test_min_direction_objective():
    # max_drawdown 为负; 但选 avg_holding_days(min 方向) 验证方向反转
    def score(p):
        return _FakeResult(stats={"avg_holding_days": p["ma_proximity"] * 100})
    out = _optimizer(score).optimize(_cfg(objective="avg_holding_days"))
    # min 方向 -> 最小 avg_holding_days (0.01x100=1) 最优
    assert out["best_params"] == {"ma_proximity": 0.01}


def test_none_and_error_results_sink_to_bottom():
    # ma_proximity=0.02 的组返回 error, 0.03 的 sortino=None -> 都应排在有效结果之后
    def score(p):
        if p["ma_proximity"] == 0.02:
            return _FakeResult(stats={}, error="boom")
        if p["ma_proximity"] == 0.03:
            return _FakeResult(stats={"sortino": None})
        return _FakeResult(stats={"sortino": 5.0})
    out = _optimizer(score).optimize(_cfg())
    assert out["best_params"] == {"ma_proximity": 0.01}
    assert out["best_score"] == 5.0
    # 失败/None 组仍在结果里但 rank 靠后
    assert out["n_completed"] == 3
    assert out["results"][0]["params"] == {"ma_proximity": 0.01}


def test_cancel_event_stops_remaining():
    ev = threading.Event()
    ev.set()  # 一开始就取消

    def score(p):
        return _FakeResult(stats={"sortino": 1.0})
    opt = _optimizer(score)
    out = opt.optimize(_cfg(), cancel_event=ev)
    # 取消后所有组跳过 -> 无有效结果
    assert opt.service.calls == []
    assert out["best_params"] is None


def test_progress_callback_reports_done_total():
    seen = []

    def score(p):
        return _FakeResult(stats={"sortino": 1.0})

    def cb(msg):
        seen.append(msg)
    _optimizer(score).optimize(_cfg(), progress_cb=cb)
    assert len(seen) == 3
    assert seen[-1]["done"] == 3
    assert all(m["total"] == 3 for m in seen)


def test_invalid_objective_rejected():
    def score(p):
        return _FakeResult(stats={"sortino": 1.0})
    with pytest.raises(ValueError, match="不支持的优化目标"):
        _optimizer(score).optimize(_cfg(objective="not_a_metric"))

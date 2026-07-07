"""优化器 API job_key 契约测试 — 守护 stream 与 cancel 的 key 对齐 (仿 PR3 C1 教训)。"""
from __future__ import annotations

from urllib.parse import urlencode

from app.api.backtest import _OPT_BT_FIELDS, _make_opt_job_key, _opt_backtest_kwargs


def _sig(bt: dict) -> str:
    return "|".join(f"{k}={bt[k]}" for k in _OPT_BT_FIELDS)


def test_job_key_deterministic():
    bt = _opt_backtest_kwargs("open_t+1", 0.0002, None, None, 5.0, 10, 1.0, 1e6, "equal", "position", 5)
    sig = _sig(bt)
    k1 = _make_opt_job_key("s", None, None, None, '{"p":[1,2]}', "sortino", None, sig)
    k2 = _make_opt_job_key("s", None, None, None, '{"p":[1,2]}', "sortino", None, sig)
    assert k1 == k2


def test_job_key_distinguishes_grid_and_objective():
    bt = _opt_backtest_kwargs("open_t+1", 0.0002, None, None, 5.0, 10, 1.0, 1e6, "equal", "position", 5)
    sig = _sig(bt)
    base = _make_opt_job_key("s", None, None, None, '{"p":[1,2]}', "sortino", None, sig)
    assert base != _make_opt_job_key("s", None, None, None, '{"p":[1,3]}', "sortino", None, sig)  # grid 不同
    assert base != _make_opt_job_key("s", None, None, None, '{"p":[1,2]}', "sharpe", None, sig)   # objective 不同


def test_stream_and_cancel_compute_same_key():
    """cancel 从 query string 复原参数, 必须与 stream 算出同一 job_key, 否则取消失效。"""
    grid = '{"ma_proximity":[0.01,0.02]}'
    # stream 侧
    bt = _opt_backtest_kwargs("open_t+1", 0.0002, None, None, 5.0, 10, 1.0, 1_000_000.0, "equal", "position", 5)
    stream_key = _make_opt_job_key("s", None, None, None, grid, "sortino", None, _sig(bt))

    # cancel 侧: 复原自 query string (urlencode 等价前端 URLSearchParams)
    qs = urlencode({
        "strategy_id": "s", "param_grid": grid, "objective": "sortino",
        "matching": "open_t+1", "fees_pct": "0.0002", "slippage_bps": "5.0",
        "max_positions": "10", "max_exposure_pct": "1.0", "initial_capital": "1000000.0",
        "position_sizing": "equal", "mode": "position", "holding_days": "5",
    })
    from urllib.parse import parse_qs
    p = parse_qs(qs)
    def g(k, d=""):
        return p.get(k, [d])[0]
    def gf(k):
        v = g(k)
        return float(v) if v else None
    bt2 = _opt_backtest_kwargs(
        g("matching", "open_t+1"), float(g("fees_pct", "0.0002")), gf("commission_pct"), gf("stamp_tax_pct"),
        float(g("slippage_bps", "5")), int(g("max_positions", "10")), float(g("max_exposure_pct", "1")),
        float(g("initial_capital", "1000000")), g("position_sizing", "equal"), g("mode", "position"),
        int(g("holding_days", "5")),
    )
    cancel_key = _make_opt_job_key(
        g("strategy_id"), g("symbols") or None, g("start") or None, g("end") or None,
        g("param_grid") or None, g("objective", "sortino"), g("direction") or None, _sig(bt2),
    )
    assert stream_key == cancel_key

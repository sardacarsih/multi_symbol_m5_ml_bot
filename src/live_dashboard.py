from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque


ASCII_BANNER = r"""
 _____              _ _             ____        _   __  __ _
|_   _| __ __ _  __| (_)_ __   __ _| __ )  ___ | |_|  \/  | |
  | || '__/ _` |/ _` | | '_ \ / _` |  _ \ / _ \| __| |\/| | |
  | || | | (_| | (_| | | | | | (_| | |_) | (_) | |_| |  | | |___
  |_||_|  \__,_|\__,_|_|_| |_|\__, |____/ \___/ \__|_|  |_|_____|
                              |___/
""".strip("\n")


@dataclass
class AccountSnapshot:
    balance: float | None = None
    equity: float | None = None
    margin_free: float | None = None
    margin_level: float | None = None
    daily_pnl: float | None = None


@dataclass
class MarketSnapshot:
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    spread_points: float | None = None
    atr: float | None = None
    last_candle_time: str = "N/A"


@dataclass
class SignalSnapshot:
    side: str = "NO_TRADE"
    confidence: float | None = None
    prob_buy: float | None = None
    prob_sell: float | None = None
    reason: str = "N/A"
    feature_time: str = "N/A"
    timeframe: str = "M5"
    lot: float = 0.0
    sl: float | None = None
    tp: float | None = None
    model_version: str = "unknown"
    gate_status: str = "missing"


@dataclass
class PositionSnapshot:
    ticket: str = "-"
    side: str = "FLAT"
    confidence: str = "-"
    lots: str = "-"
    open_price: str = "-"
    sl: str = "-"
    tp: str = "-"
    pnl: str = "-"


@dataclass
class LiveSnapshot:
    symbol: str
    mt5_symbol: str
    mode: str
    strategy: str
    htf: str
    connected: bool
    heartbeat: str
    trade_allowed: str
    updated: str
    open_positions_symbol: int = 0
    open_positions_total: int = 0
    account: AccountSnapshot = field(default_factory=AccountSnapshot)
    market: MarketSnapshot = field(default_factory=MarketSnapshot)
    signal: SignalSnapshot = field(default_factory=SignalSnapshot)
    positions: list[PositionSnapshot] = field(default_factory=list)
    trend: dict[str, str] = field(default_factory=lambda: {"M5": "N/A", "M15": "N/A", "H1": "N/A", "H4": "N/A"})
    error: str | None = None


def _number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def account_snapshot(mt5) -> AccountSnapshot:
    account_info = getattr(mt5, "account_info", None)
    account = account_info() if callable(account_info) else None
    if account is None:
        return AccountSnapshot()
    return AccountSnapshot(
        balance=_number(getattr(account, "balance", None)),
        equity=_number(getattr(account, "equity", None)),
        margin_free=_number(getattr(account, "margin_free", None)),
        margin_level=_number(getattr(account, "margin_level", None)),
    )


def positions_snapshot(mt5, mt5_symbol: str, magic_number: int) -> list[PositionSnapshot]:
    positions_get = getattr(mt5, "positions_get", None)
    positions = positions_get() if callable(positions_get) else None
    if not positions:
        return []
    result = []
    for position in positions:
        if getattr(position, "magic", None) != magic_number:
            continue
        if getattr(position, "symbol", None) != mt5_symbol:
            continue
        pos_type = getattr(position, "type", None)
        side = "BUY" if pos_type == 0 else "SELL" if pos_type == 1 else str(pos_type or "OPEN")
        result.append(
            PositionSnapshot(
                ticket=str(getattr(position, "ticket", getattr(position, "identifier", "-"))),
                side=side,
                lots=_format_number(getattr(position, "volume", None), 2),
                open_price=_format_number(getattr(position, "price_open", None), 2),
                sl=_format_number(getattr(position, "sl", None), 2),
                tp=_format_number(getattr(position, "tp", None), 2),
                pnl=_format_number(getattr(position, "profit", None), 2),
            )
        )
    return result


def trend_snapshot(latest) -> dict[str, str]:
    def from_pair(close_col: str, ema_col: str) -> str:
        close = _number(latest.get(close_col))
        ema = _number(latest.get(ema_col))
        if close is None or ema is None:
            return "N/A"
        if close > ema:
            return "UP"
        if close < ema:
            return "DOWN"
        return "WAIT"

    def from_value(column: str) -> str:
        value = _number(latest.get(column))
        if value is None:
            return "N/A"
        if value > 0:
            return "UP"
        if value < 0:
            return "DOWN"
        return "WAIT"

    return {
        "M5": from_pair("close", "ema_20"),
        "M15": from_value("m15_close_to_ema20"),
        "H1": from_value("h1_close_to_ema20"),
        "H4": "N/A",
    }


def _format_number(value: Any, digits: int = 2, default: str = "N/A") -> str:
    number = _number(value)
    if number is None:
        return default
    return f"{number:,.{digits}f}"


class RichDashboard:
    def __init__(self, focus_symbol: str, symbols: list[str], log_limit: int = 100):
        from rich.console import Console

        self.console = Console()
        self.focus_symbol = focus_symbol
        self.symbols = symbols
        self.snapshots: dict[str, LiveSnapshot] = {}
        self.main_events: Deque[str] = deque(maxlen=log_limit)
        self.signal_events: Deque[str] = deque(maxlen=log_limit)
        self.trade_events: Deque[str] = deque(maxlen=log_limit)
        self.error_events: Deque[str] = deque(maxlen=log_limit)

    def set_snapshot(self, snapshot: LiveSnapshot) -> None:
        self.snapshots[snapshot.symbol] = snapshot
        self.focus_symbol = self.focus_symbol if self.focus_symbol in self.snapshots else snapshot.symbol
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.main_events.append(f"{timestamp} INFO {snapshot.symbol} {snapshot.signal.side} {snapshot.signal.reason}")
        self.signal_events.append(
            f"{timestamp} {snapshot.symbol} buy={_format_number(snapshot.signal.prob_buy, 3)} "
            f"sell={_format_number(snapshot.signal.prob_sell, 3)}"
        )
        if snapshot.signal.reason in {"READY", "DRY_RUN"} and snapshot.signal.side != "NO_TRADE":
            self.trade_events.append(f"{timestamp} {snapshot.symbol} {snapshot.signal.side} lot={snapshot.signal.lot:.2f}")
        if snapshot.error:
            self.error_events.append(f"{timestamp} ERROR {snapshot.symbol} {snapshot.error}")

    def add_error(self, symbol: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.error_events.append(f"{timestamp} ERROR {symbol} {message}")
        self.main_events.append(f"{timestamp} ERROR {symbol} evaluation failed")

    def render(self):
        from rich.align import Align
        from rich.layout import Layout
        from rich.panel import Panel

        snapshot = self.snapshots.get(self.focus_symbol) or next(iter(self.snapshots.values()), None)
        layout = Layout()
        layout.split_column(
            Layout(self._header(snapshot), size=12),
            Layout(name="body", ratio=3),
            Layout(name="logs", ratio=1),
        )
        layout["body"].split_row(
            Layout(name="left", size=50),
            Layout(self._positions(snapshot), ratio=2),
            Layout(name="right", size=50),
        )
        layout["left"].split_column(
            Layout(self._status(snapshot), ratio=1),
            Layout(self._account(snapshot), ratio=1),
        )
        layout["right"].split_column(
            Layout(self._market(snapshot), ratio=1),
            Layout(self._trend(snapshot), ratio=1),
            Layout(self._signal(snapshot), ratio=1),
        )
        layout["logs"].split_row(
            Layout(self._events_panel("Main", self.main_events, "blue")),
            Layout(self._events_panel("Signals", self.signal_events, "yellow")),
            Layout(self._events_panel("Trades", self.trade_events, "green")),
            Layout(self._events_panel("Errors", self.error_events, "red")),
        )
        return Align.center(layout, vertical="top")

    def _header(self, snapshot: LiveSnapshot | None):
        from rich.align import Align
        from rich.panel import Panel
        from rich.text import Text

        symbol = snapshot.symbol if snapshot else self.focus_symbol
        title = Text()
        title.append("=" * 96 + "\n", style="cyan")
        title.append(ASCII_BANNER + "\n", style="bold cyan")
        title.append(f"[{symbol} M5 | Machine Learning Live Engine]\n", style="bold cyan")
        title.append("=" * 96, style="cyan")
        return Panel(Align.center(title), border_style="bright_blue")

    def _status(self, snapshot: LiveSnapshot | None):
        rows = [
            ("Symbol", snapshot.symbol if snapshot else self.focus_symbol),
            ("Mode", snapshot.mode if snapshot else "N/A"),
            ("Strategy", snapshot.strategy if snapshot else "N/A"),
            ("HTF", snapshot.htf if snapshot else "N/A"),
            ("Connection", "CONNECTED" if snapshot and snapshot.connected else "N/A"),
            ("Heartbeat", snapshot.heartbeat if snapshot else "N/A"),
            ("Trade Allowed", snapshot.trade_allowed if snapshot else "N/A"),
            ("Updated", snapshot.updated if snapshot else "N/A"),
        ]
        return self._kv_panel("Status", rows, "bright_blue")

    def _account(self, snapshot: LiveSnapshot | None):
        account = snapshot.account if snapshot else AccountSnapshot()
        rows = [
            ("Balance", _format_number(account.balance)),
            ("Equity", _format_number(account.equity)),
            ("Free Margin", _format_number(account.margin_free)),
            ("Margin Level", _format_number(account.margin_level, 1)),
            ("Daily PnL", _format_number(account.daily_pnl)),
        ]
        return self._kv_panel("Account", rows, "blue")

    def _market(self, snapshot: LiveSnapshot | None):
        market = snapshot.market if snapshot else MarketSnapshot()
        rows = [
            ("Bid", _format_number(market.bid)),
            ("Ask", _format_number(market.ask)),
            ("Spread", _format_number(market.spread_points, 1, "-") + " pts"),
            ("Last Candle", market.last_candle_time),
            ("ATR", _format_number(market.atr)),
        ]
        return self._kv_panel("Market", rows, "bright_blue")

    def _trend(self, snapshot: LiveSnapshot | None):
        trend = snapshot.trend if snapshot else {}
        return self._kv_panel("Trend", [(tf, trend.get(tf, "N/A")) for tf in ["M5", "M15", "H1", "H4"]], "magenta")

    def _signal(self, snapshot: LiveSnapshot | None):
        signal = snapshot.signal if snapshot else SignalSnapshot()
        rows = [
            ("Action", signal.side),
            ("Time", signal.feature_time),
            ("TF", signal.timeframe),
            ("Confidence", _format_number(signal.confidence, 3)),
            ("Reason", signal.reason),
        ]
        return self._kv_panel("Signal", rows, "cyan")

    def _positions(self, snapshot: LiveSnapshot | None):
        from rich.panel import Panel
        from rich.table import Table

        table = Table(expand=True)
        for column in ["Ticket", "Side", "Confidence", "Lots", "Open", "SL", "TP", "PnL"]:
            table.add_column(column)
        positions = snapshot.positions if snapshot else []
        if not positions:
            table.add_row("-", "FLAT", "-", "-", "-", "-", "-", "-")
        for position in positions:
            table.add_row(
                position.ticket,
                position.side,
                position.confidence,
                position.lots,
                position.open_price,
                position.sl,
                position.tp,
                position.pnl,
            )
        return Panel(table, title="Open Positions", border_style="green")

    def _events_panel(self, title: str, events: Deque[str], style: str):
        from rich.panel import Panel
        from rich.text import Text

        text = Text("\n".join(events) if events else "No events yet", style="white" if events else "dim")
        return Panel(text, title=title, border_style=style)

    def _kv_panel(self, title: str, rows: list[tuple[str, str]], style: str):
        from rich.panel import Panel
        from rich.table import Table

        table = Table.grid(expand=True)
        table.add_column(style="dim bold")
        table.add_column(justify="right")
        for key, value in rows:
            table.add_row(key, value)
        return Panel(table, title=title, border_style=style)

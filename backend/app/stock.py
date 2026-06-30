from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

try:
    import yfinance as yf

    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False
    logger.info("yfinance not installed — stock chart generation disabled")

PERIOD_CHOICES = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
INTERVAL_CHOICES = ["1d", "1wk", "1mo"]
CHART_TYPE_CHOICES = ["candle", "line", "infographic"]
INDICATOR_CHOICES = ["sma20", "sma60", "ema20", "bollinger", "rsi", "macd", "volume"]

UP_COLOR = "#2e7d32"
DOWN_COLOR = "#d32f2f"
THEME_BG = "#0f172a"
THEME_PANEL = "#162033"
THEME_GRID = "#27344a"
THEME_TEXT = "#e5e7eb"
THEME_MUTED = "#94a3b8"
GRADIENT_COLORS = ["#4285f4", "#9b72cb", "#d96570"]


class StockChartRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=20)
    period: str = Field(default="6mo")
    interval: str = Field(default="1d")
    chart_type: str = Field(default="candle")
    indicators: list[str] = Field(default_factory=lambda: ["sma20", "volume"])

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("period")
    @classmethod
    def validate_period(cls, value: str) -> str:
        if value not in PERIOD_CHOICES:
            raise ValueError(f"period must be one of {PERIOD_CHOICES}")
        return value

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, value: str) -> str:
        if value not in INTERVAL_CHOICES:
            raise ValueError(f"interval must be one of {INTERVAL_CHOICES}")
        return value

    @field_validator("chart_type")
    @classmethod
    def validate_chart_type(cls, value: str) -> str:
        if value not in CHART_TYPE_CHOICES:
            raise ValueError(f"chart_type must be one of {CHART_TYPE_CHOICES}")
        return value

    @field_validator("indicators")
    @classmethod
    def validate_indicators(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(INDICATOR_CHOICES))
        if unknown:
            raise ValueError(f"unknown indicators: {unknown}")
        return value


class StockChartResult(BaseModel):
    file_url: str
    media_type: str
    ticker: str
    period: str
    interval: str
    chart_type: str
    data_url: str | None = None
    summary: dict[str, float]


class StockChartService:
    def __init__(self, output_dir: Path, data_fetcher=None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._data_fetcher = data_fetcher or self._fetch_yfinance

    @staticmethod
    def available_periods() -> list[str]:
        return PERIOD_CHOICES

    @staticmethod
    def available_intervals() -> list[str]:
        return INTERVAL_CHOICES

    @staticmethod
    def available_chart_types() -> list[str]:
        return CHART_TYPE_CHOICES

    @staticmethod
    def available_indicators() -> list[str]:
        return INDICATOR_CHOICES

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        if not _YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance가 설치되지 않아 실시간 시세를 가져올 수 없습니다.")
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            raise ValueError(f"'{ticker}' 종목의 시세 데이터를 찾을 수 없습니다.")
        return df

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
        macd_line = close.ewm(span=fast).mean() - close.ewm(span=slow).mean()
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, request: StockChartRequest) -> StockChartResult:
        df = self._data_fetcher(request.ticker, request.period, request.interval)
        summary = self._build_summary(df)

        file_name = f"stock_{uuid.uuid4().hex[:8]}.png"
        output_path = self.output_dir / file_name

        if request.chart_type == "infographic":
            fig = self._render_infographic(df, request, summary)
        else:
            fig = self._render_chart(df, request)

        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        encoded = base64.b64encode(output_path.read_bytes()).decode("ascii")
        return StockChartResult(
            file_url=f"/outputs/{file_name}",
            media_type="image/png",
            ticker=request.ticker,
            period=request.period,
            interval=request.interval,
            chart_type=request.chart_type,
            data_url=f"data:image/png;base64,{encoded}",
            summary=summary,
        )

    @staticmethod
    def _build_summary(df: pd.DataFrame) -> dict[str, float]:
        close = df["Close"]
        return {
            "last_close": round(float(close.iloc[-1]), 2),
            "change_pct": round(float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2),
            "high": round(float(df["High"].max()), 2),
            "low": round(float(df["Low"].min()), 2),
            "avg_volume": round(float(df["Volume"].mean()), 0),
        }

    # ------------------------------------------------------------------
    # Rendering — candle / line chart with indicator overlays
    # ------------------------------------------------------------------

    def _render_chart(self, df: pd.DataFrame, request: StockChartRequest) -> plt.Figure:
        indicators = request.indicators
        show_rsi = "rsi" in indicators
        show_macd = "macd" in indicators
        show_volume = "volume" in indicators

        panel_flags = [True, show_volume, show_rsi, show_macd]
        heights = [3.2 if i == 0 else 1.0 for i, flag in enumerate(panel_flags) if flag]
        fig, axes = plt.subplots(
            nrows=sum(panel_flags),
            ncols=1,
            figsize=(11, 4 + 1.4 * (sum(panel_flags) - 1)),
            sharex=True,
            gridspec_kw={"height_ratios": heights, "hspace": 0.08},
            facecolor=THEME_BG,
        )
        axes = [axes] if sum(panel_flags) == 1 else list(axes)
        axis_iter = iter(axes)
        price_ax = next(axis_iter)
        self._style_axis(price_ax)

        x = mdates.date2num(df.index.to_pydatetime())
        if request.chart_type == "candle":
            self._draw_candlesticks(price_ax, x, df)
        else:
            price_ax.plot(x, df["Close"], color="#4285f4", linewidth=1.6, label="Close")

        if "sma20" in indicators:
            price_ax.plot(x, df["Close"].rolling(20).mean(), color="#f4b400", linewidth=1.2, label="SMA20")
        if "sma60" in indicators:
            price_ax.plot(x, df["Close"].rolling(60).mean(), color="#9b72cb", linewidth=1.2, label="SMA60")
        if "ema20" in indicators:
            price_ax.plot(x, df["Close"].ewm(span=20).mean(), color="#34a853", linewidth=1.2, label="EMA20")
        if "bollinger" in indicators:
            mid = df["Close"].rolling(20).mean()
            std = df["Close"].rolling(20).std()
            price_ax.plot(x, mid + 2 * std, color=THEME_MUTED, linewidth=1, linestyle="--", label="BOLL Upper")
            price_ax.plot(x, mid - 2 * std, color=THEME_MUTED, linewidth=1, linestyle="--", label="BOLL Lower")

        price_ax.set_title(
            f"{request.ticker}  ·  {request.period} / {request.interval}",
            color=THEME_TEXT,
            fontsize=14,
            loc="left",
            fontweight="bold",
        )
        price_ax.legend(loc="upper left", fontsize=8, facecolor=THEME_PANEL, edgecolor=THEME_GRID, labelcolor=THEME_TEXT)

        if show_volume:
            vol_ax = next(axis_iter)
            self._style_axis(vol_ax)
            colors = [UP_COLOR if c >= o else DOWN_COLOR for o, c in zip(df["Open"], df["Close"])]
            vol_ax.bar(x, df["Volume"], color=colors, width=0.6 * (x[1] - x[0]) if len(x) > 1 else 0.6)
            vol_ax.set_ylabel("Volume", color=THEME_MUTED, fontsize=9)

        if show_rsi:
            rsi_ax = next(axis_iter)
            self._style_axis(rsi_ax)
            rsi_ax.plot(x, self._rsi(df["Close"]), color="#34a853", linewidth=1.2)
            rsi_ax.axhline(70, color=THEME_MUTED, linewidth=0.8, linestyle="--")
            rsi_ax.axhline(30, color=THEME_MUTED, linewidth=0.8, linestyle="--")
            rsi_ax.set_ylabel("RSI", color=THEME_MUTED, fontsize=9)
            rsi_ax.set_ylim(0, 100)

        if show_macd:
            macd_ax = next(axis_iter)
            self._style_axis(macd_ax)
            macd_line, signal_line = self._macd(df["Close"])
            macd_ax.plot(x, macd_line, color="#4285f4", linewidth=1.2, label="MACD")
            macd_ax.plot(x, signal_line, color="#d96570", linewidth=1.2, label="Signal")
            macd_ax.set_ylabel("MACD", color=THEME_MUTED, fontsize=9)
            macd_ax.legend(loc="upper left", fontsize=7, facecolor=THEME_PANEL, edgecolor=THEME_GRID, labelcolor=THEME_TEXT)

        axes[-1].xaxis_date()
        fig.autofmt_xdate()
        return fig

    @staticmethod
    def _draw_candlesticks(ax: plt.Axes, x, df: pd.DataFrame) -> None:
        width = 0.6 * (x[1] - x[0]) if len(x) > 1 else 0.6
        for xi, (_, row) in zip(x, df.iterrows()):
            color = UP_COLOR if row["Close"] >= row["Open"] else DOWN_COLOR
            ax.add_line(Line2D([xi, xi], [row["Low"], row["High"]], color=color, linewidth=0.8))
            lower = min(row["Open"], row["Close"])
            height = abs(row["Close"] - row["Open"]) or 0.01
            ax.add_patch(Rectangle((xi - width / 2, lower), width, height, facecolor=color, edgecolor=color))
        ax.set_xlim(x[0] - width * 2, x[-1] + width * 2)

    @staticmethod
    def _style_axis(ax: plt.Axes) -> None:
        ax.set_facecolor(THEME_PANEL)
        ax.grid(True, color=THEME_GRID, linewidth=0.6, alpha=0.6)
        ax.tick_params(colors=THEME_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(THEME_GRID)

    # ------------------------------------------------------------------
    # Rendering — infographic / poster-style summary screen
    # ------------------------------------------------------------------

    def _render_infographic(
        self, df: pd.DataFrame, request: StockChartRequest, summary: dict[str, float]
    ) -> plt.Figure:
        fig = plt.figure(figsize=(9, 12), facecolor=THEME_BG)
        grid = fig.add_gridspec(nrows=5, ncols=2, height_ratios=[1.1, 0.8, 2.0, 1.1, 1.1], hspace=0.55, wspace=0.3)

        is_up = summary["change_pct"] >= 0
        accent = UP_COLOR if is_up else DOWN_COLOR
        arrow = "▲" if is_up else "▼"

        header_ax = fig.add_subplot(grid[0, :])
        header_ax.axis("off")
        for i, color in enumerate(GRADIENT_COLORS):
            header_ax.add_patch(Rectangle((i / 3, 0), 1 / 3, 0.06, transform=header_ax.transAxes, color=color, clip_on=False))
        header_ax.text(0, 0.55, request.ticker, fontsize=34, fontweight="bold", color=THEME_TEXT, transform=header_ax.transAxes)
        header_ax.text(
            0,
            0.2,
            f"{request.period.upper()} ANALYSIS · {request.interval}",
            fontsize=12,
            color=THEME_MUTED,
            transform=header_ax.transAxes,
        )

        price_ax = fig.add_subplot(grid[1, :])
        price_ax.axis("off")
        price_ax.text(0, 0.5, f"{summary['last_close']:,.2f}", fontsize=30, fontweight="bold", color=THEME_TEXT, transform=price_ax.transAxes)
        price_ax.text(
            0.45,
            0.55,
            f"{arrow} {summary['change_pct']:+.2f}%",
            fontsize=20,
            fontweight="bold",
            color=accent,
            transform=price_ax.transAxes,
        )

        chart_ax = fig.add_subplot(grid[2, :])
        self._style_axis(chart_ax)
        chart_ax.set_facecolor(THEME_BG)
        x = mdates.date2num(df.index.to_pydatetime())
        chart_ax.plot(x, df["Close"], color=accent, linewidth=2)
        chart_ax.fill_between(x, df["Close"], df["Close"].min(), color=accent, alpha=0.15)
        chart_ax.plot(x, df["Close"].rolling(20).mean(), color=THEME_MUTED, linewidth=1, linestyle="--")
        chart_ax.xaxis_date()
        chart_ax.tick_params(labelsize=8)

        metrics = [
            ("PERIOD HIGH", f"{summary['high']:,.2f}"),
            ("PERIOD LOW", f"{summary['low']:,.2f}"),
            ("AVG VOLUME", f"{summary['avg_volume']:,.0f}"),
            ("PERIOD CHANGE", f"{summary['change_pct']:+.2f}%"),
        ]
        for idx, (label, value) in enumerate(metrics):
            row = 3 + idx // 2
            col = idx % 2
            ax = fig.add_subplot(grid[row, col])
            ax.axis("off")
            ax.add_patch(
                Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=THEME_PANEL, edgecolor=THEME_GRID, clip_on=False)
            )
            ax.text(0.08, 0.62, label, fontsize=11, color=THEME_MUTED, transform=ax.transAxes)
            ax.text(0.08, 0.22, value, fontsize=18, fontweight="bold", color=THEME_TEXT, transform=ax.transAxes)

        fig.suptitle("")
        return fig

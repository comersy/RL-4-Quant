import sys

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from envs.pricing import black_scholes
from envs.underlying import Underlying

BG = "#ffffff"
BG2 = "#f5f5f5"
BORDER = "#dddddd"
GREEN = "#00aa66"
RED = "#dd2244"
FG = "#111111"
FG2 = "#666666"
MONO = "Courier"


class TradingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL-4-Quant // Trading Desk")
        self.resize(1200, 820)

        self.S0 = 100.0
        self.daily_vol = 0.01
        self.r = 0.0
        self.underlying = None
        self.history_prices = []
        self.live_prices = []
        self.portfolio = []
        self.pnl_history = [0.0]
        self.total_pnl = 0.0
        self.log = []
        self.trading_day = 0

        self._build_ui()
        self._new_episode()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        root.setStyleSheet(self._stylesheet())
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 8, 16, 16)
        root_layout.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(10)
        title = QLabel("RL-4-Quant // Trading Desk")
        title.setFont(QFont(MONO, 15, QFont.Bold))
        top.addWidget(title, stretch=1)
        self.lbl_day = self._metric(top, "DAY", "0")
        self.lbl_spot = self._metric(top, "SPOT", "-")
        self.lbl_vol = self._metric(top, "REAL VOL", "-")
        self.lbl_pnl = self._metric(top, "P&L", "0.00", color=GREEN)
        root_layout.addLayout(top)

        main = QHBoxLayout()
        main.setSpacing(12)
        root_layout.addLayout(main, stretch=1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = plt.Figure(figsize=(7, 5.5), facecolor=BG)
        gs = gridspec.GridSpec(2, 1, figure=self.fig, hspace=0.45, height_ratios=[2, 1])
        self.ax_price = self.fig.add_subplot(gs[0])
        self.ax_pnl = self.fig.add_subplot(gs[1])
        self._style_axes()

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.canvas)
        main.addWidget(left, stretch=1)

        right = QWidget()
        right.setFixedWidth(290)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self._build_controls(right_layout)
        main.addWidget(right)

    def _stylesheet(self):
        return f"""
        QWidget#root {{
            background: {BG};
            color: {FG};
            font-family: {MONO};
        }}
        QLabel {{
            color: {FG};
            background: transparent;
        }}
        QLineEdit {{
            background: {BG};
            color: {FG};
            border: 1px solid {BORDER};
            padding: 4px 6px;
            selection-background-color: {GREEN};
        }}
        QPushButton {{
            border: none;
            padding: 7px 8px;
            font-weight: bold;
        }}
        QTextEdit {{
            background: {BG2};
            color: {FG2};
            border: none;
        }}
        QScrollArea {{
            border: none;
            background: {BG2};
        }}
        """

    def _metric(self, parent, label, value, color=FG):
        frame = QFrame()
        frame.setStyleSheet(f"background: {BG2}; border: none;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(1)

        name = QLabel(label)
        name.setAlignment(Qt.AlignCenter)
        name.setStyleSheet(f"color: {FG2};")
        name.setFont(QFont(MONO, 7))
        layout.addWidget(name)

        val = QLabel(value)
        val.setAlignment(Qt.AlignCenter)
        val.setStyleSheet(f"color: {color};")
        val.setFont(QFont(MONO, 12, QFont.Bold))
        layout.addWidget(val)

        parent.addWidget(frame)
        return val

    def _build_controls(self, parent):
        self._sep(parent, "CONFIG")
        cfg = self._panel()
        cfg_layout = QVBoxLayout(cfg)
        cfg_layout.setContentsMargins(8, 8, 8, 8)
        self.s0_var = self._entry_row(cfg_layout, "S0", 100.0)
        self.vol_var = self._entry_row(cfg_layout, "Daily Vol", 0.01)
        cfg_layout.addWidget(self._btn("New Episode", FG, self._new_episode))
        parent.addWidget(cfg)

        self._sep(parent, "PLACE TRADE")
        trade = self._panel()
        trade_layout = QVBoxLayout(trade)
        trade_layout.setContentsMargins(8, 8, 8, 8)
        self.strike_var = self._entry_row(trade_layout, "Strike K", 100.0)
        self.mat_var = self._entry_row(trade_layout, "Maturity (d)", 30)
        self.qty_var = self._entry_row(trade_layout, "Quantity", 1)

        prices = QGridLayout()
        prices.setContentsMargins(0, 4, 0, 4)
        prices.addWidget(self._muted_label("CALL", 8), 0, 0)
        self.lbl_call = QLabel("-")
        self.lbl_call.setStyleSheet(f"color: {GREEN}; font-weight: bold;")
        prices.addWidget(self.lbl_call, 0, 1, alignment=Qt.AlignRight)
        prices.addWidget(self._muted_label("PUT", 8), 1, 0)
        self.lbl_put = QLabel("-")
        self.lbl_put.setStyleSheet(f"color: {RED}; font-weight: bold;")
        prices.addWidget(self.lbl_put, 1, 1, alignment=Qt.AlignRight)
        trade_layout.addLayout(prices)

        self.strike_var.textChanged.connect(self._update_prices)
        self.mat_var.textChanged.connect(self._update_prices)

        buttons = QGridLayout()
        buttons.setSpacing(4)
        buttons.addWidget(self._btn("BUY CALL", GREEN, lambda: self._order("call", False)), 0, 0)
        buttons.addWidget(self._btn("BUY PUT", RED, lambda: self._order("put", False)), 0, 1)
        buttons.addWidget(self._btn("SELL CALL", GREEN, lambda: self._order("call", True), outline=True), 1, 0)
        buttons.addWidget(self._btn("SELL PUT", RED, lambda: self._order("put", True), outline=True), 1, 1)
        trade_layout.addLayout(buttons)
        parent.addWidget(trade)

        self._sep(parent, "NAVIGATE")
        nav = self._panel()
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(8, 8, 8, 8)
        nav_layout.addWidget(self._btn("Next Day", FG, self._next_day))
        nav_layout.addWidget(self._btn("Skip Week", FG2, self._skip_week))
        parent.addWidget(nav)

        self._sep(parent, "PORTFOLIO")
        self.port_frame = self._panel()
        self.port_layout = QVBoxLayout(self.port_frame)
        self.port_layout.setContentsMargins(4, 4, 4, 4)
        self.port_layout.setSpacing(2)

        portfolio_scroll = QScrollArea()
        portfolio_scroll.setWidgetResizable(True)
        portfolio_scroll.setFixedHeight(120)
        portfolio_scroll.setWidget(self.port_frame)
        parent.addWidget(portfolio_scroll)

        self._sep(parent, "TRADE LOG")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont(MONO, 7))
        parent.addWidget(self.log_text, stretch=1)

    def _panel(self):
        panel = QFrame()
        panel.setStyleSheet(f"background: {BG2}; border: none;")
        return panel

    def _muted_label(self, text, size=7):
        label = QLabel(text)
        label.setStyleSheet(f"color: {FG2};")
        label.setFont(QFont(MONO, size))
        return label

    def _sep(self, parent, title):
        label = self._muted_label(title, 7)
        label.setContentsMargins(4, 10, 0, 1)
        parent.addWidget(label)

    def _entry_row(self, parent, label, default):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        name = self._muted_label(label, 8)
        name.setFixedWidth(120)
        row.addWidget(name)

        entry = QLineEdit(str(default))
        entry.setFont(QFont(MONO, 9))
        entry.setAlignment(Qt.AlignRight)
        row.addWidget(entry)

        parent.addLayout(row)
        return entry

    def _btn(self, text, color, cmd, outline=False):
        button = QPushButton(text)
        button.setCursor(Qt.PointingHandCursor)
        button.setFont(QFont(MONO, 8, QFont.Bold))
        if outline:
            button.setStyleSheet(
                f"background: {BG2}; color: {color}; border: 1px solid {color};"
            )
        else:
            button.setStyleSheet(f"background: {color}; color: {BG};")
        button.clicked.connect(cmd)
        return button

    def _new_episode(self):
        try:
            self.S0 = float(self.s0_var.text())
            self.daily_vol = float(self.vol_var.text())
        except ValueError:
            pass
        self.underlying = Underlying(S0=self.S0, daily_vol=self.daily_vol)
        self.underlying.simulate(251)
        self.history_prices = list(self.underlying.prices)
        self.live_prices = [self.history_prices[-1]]
        self.trading_day = 0
        self.portfolio = []
        self.pnl_history = [0.0]
        self.total_pnl = 0.0
        self.log = []
        self._refresh()

    def _current_price(self):
        return self.live_prices[-1]

    def _all_prices(self):
        return self.history_prices + self.live_prices[1:]

    def _realized_vol(self):
        all_p = self._all_prices()
        if len(all_p) < 2:
            return 0.01
        lr = np.diff(np.log(all_p))
        return max(float(np.std(lr)) * np.sqrt(252), 0.001)

    def _order(self, option_type, is_short):
        try:
            K = float(self.strike_var.text())
            mat = int(self.mat_var.text())
            qty = int(self.qty_var.text())
        except ValueError:
            return
        if qty <= 0 or mat <= 0:
            return

        S = self._current_price()
        sigma = self._realized_vol()
        T = mat / 252
        price = black_scholes(S=S, K=K, T=T, r=self.r, sigma=sigma, option_type=option_type)
        direction = "SELL" if is_short else "BUY "
        self.portfolio.append(
            {
                "option_type": option_type,
                "strike": K,
                "maturite": mat,
                "quantite": qty,
                "is_short": is_short,
                "entry_day": self.trading_day,
                "entry_price": price,
                "last_price": price,
            }
        )

        fees = 0.65 * qty
        self.total_pnl -= fees
        self.pnl_history[-1] = self.total_pnl
        self.log.append(
            "Day %3d | %s %4s K=%.1f T=%dd qty=%d @ %.4f  fees=-%.2f"
            % (self.trading_day, direction, option_type.upper(), K, mat, qty, price, fees)
        )
        self._refresh()

    def _next_day(self):
        self._advance()
        self._refresh()

    def _skip_week(self):
        for _ in range(5):
            self._advance()
        self._refresh()

    def _advance(self):
        S = self.live_prices[-1]
        Z = np.random.standard_normal()
        S_new = S * np.exp(-0.5 * self.daily_vol**2 + self.daily_vol * Z)
        self.live_prices.append(S_new)
        self.trading_day += 1
        self._check_early_exercise()
        self.total_pnl += self._mark_to_market()
        self.pnl_history.append(self.total_pnl)

    def _check_early_exercise(self):
        """For short positions: if ITM, the buyer exercises immediately."""
        S = self._current_price()
        for p in self.portfolio:
            if not p["is_short"]:
                continue
            if p["option_type"] == "call" and S > p["strike"]:
                payoff = (S - p["strike"]) * p["quantite"]
                loss = -payoff - (-p["last_price"] * p["quantite"])
                self.total_pnl += loss
                self.log.append(
                    f"Day {self.trading_day:>3} | EXERCISED CALL K={p['strike']:.1f} "
                    f"S={S:.2f}  loss={loss:+.4f}"
                )
                p["expired"] = True
            elif p["option_type"] == "put" and S < p["strike"]:
                payoff = (p["strike"] - S) * p["quantite"]
                loss = -payoff - (-p["last_price"] * p["quantite"])
                self.total_pnl += loss
                self.log.append(
                    f"Day {self.trading_day:>3} | EXERCISED PUT  K={p['strike']:.1f} "
                    f"S={S:.2f}  loss={loss:+.4f}"
                )
                p["expired"] = True
        self.portfolio = [p for p in self.portfolio if not p.get("expired")]

    def _mark_to_market(self):
        S = self._current_price()
        sigma = self._realized_vol()
        t = self.trading_day
        daily_pnl = 0.0
        for p in self.portfolio:
            days_left = p["entry_day"] + p["maturite"] - t
            if days_left <= 0:
                if p["option_type"] == "call":
                    payoff = max(S - p["strike"], 0.0)
                else:
                    payoff = max(p["strike"] - S, 0.0)
                direction = -1 if p["is_short"] else 1
                daily_pnl += direction * (payoff - p["last_price"]) * p["quantite"]
                p["last_price"] = payoff
                p["expired"] = True
            else:
                T = days_left / 252
                new_price = black_scholes(
                    S=S,
                    K=p["strike"],
                    T=T,
                    r=self.r,
                    sigma=sigma,
                    option_type=p["option_type"],
                )
                direction = -1 if p["is_short"] else 1
                daily_pnl += direction * (new_price - p["last_price"]) * p["quantite"]
                p["last_price"] = new_price
        self.portfolio = [p for p in self.portfolio if not p.get("expired")]
        return daily_pnl

    def _close_position(self, idx):
        if idx >= len(self.portfolio):
            return
        p = self.portfolio[idx]
        direction = "SELL" if p["is_short"] else "BUY "
        realized = (-1 if p["is_short"] else 1) * (
            p["last_price"] - p["entry_price"]
        ) * p["quantite"]
        self.log.append(
            f"Day {self.trading_day:>3} | CLOSE {direction.strip()} "
            f"{p['option_type'].upper():4} K={p['strike']:.1f} "
            f"@ {p['last_price']:.4f}  PnL={realized:+.4f}"
        )
        self.portfolio.pop(idx)
        self._refresh()

    def _update_prices(self):
        if not self.live_prices:
            return
        try:
            K = float(self.strike_var.text())
            mat = max(int(self.mat_var.text()), 1)
        except ValueError:
            return
        S = self._current_price()
        sigma = self._realized_vol()
        T = mat / 252
        call_p = black_scholes(S=S, K=K, T=T, r=self.r, sigma=sigma, option_type="call")
        put_p = black_scholes(S=S, K=K, T=T, r=self.r, sigma=sigma, option_type="put")
        self.lbl_call.setText(f"{call_p:.4f}")
        self.lbl_put.setText(f"{put_p:.4f}")

    def _refresh(self):
        S = self._current_price()
        sigma = self._realized_vol()
        pnl = self.total_pnl

        self.lbl_day.setText(str(self.trading_day))
        self.lbl_spot.setText(f"{S:.2f}")
        self.lbl_vol.setText(f"{sigma * 100:.1f}%")
        self.lbl_pnl.setText(f"{pnl:+.2f}")
        self.lbl_pnl.setStyleSheet(f"color: {GREEN if pnl >= 0 else RED};")

        all_p = self._all_prices()
        n_hist = len(self.history_prices)
        n_total = len(all_p)

        self.ax_price.cla()
        self.ax_pnl.cla()
        self._style_axes()

        self.ax_price.plot(range(n_hist), self.history_prices, color=BORDER, lw=1.5)
        if len(self.live_prices) > 1:
            live_x = list(range(n_hist - 1, n_total))
            self.ax_price.plot(
                live_x,
                [self.history_prices[-1]] + self.live_prices[1:],
                color=GREEN,
                lw=1.5,
            )
        self.ax_price.axvline(x=n_hist - 1, color=BORDER, lw=1, ls="--")
        self.ax_price.set_title("grey = history   green = trading", color=FG2, fontsize=8, pad=3)

        fill_color = GREEN if self.pnl_history[-1] >= 0 else RED
        self.ax_pnl.plot(self.pnl_history, color=fill_color, lw=1.5)
        self.ax_pnl.fill_between(
            range(len(self.pnl_history)), self.pnl_history, alpha=0.15, color=fill_color
        )
        self.ax_pnl.axhline(0, color=BORDER, lw=0.8)
        self.ax_pnl.set_title("P&L", color=FG2, fontsize=8, pad=3)
        self.canvas.draw()

        self._refresh_portfolio()
        self.log_text.setPlainText("\n".join(reversed(self.log[-30:])))
        self._update_prices()

    def _style_axes(self):
        for ax in [self.ax_price, self.ax_pnl]:
            ax.set_facecolor(BG2)
            ax.tick_params(colors=FG2, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)

    def _refresh_portfolio(self):
        while self.port_layout.count():
            item = self.port_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not self.portfolio:
            empty = self._muted_label("No open positions.", 8)
            self.port_layout.addWidget(empty)
            self.port_layout.addStretch(1)
            return

        for i, p in enumerate(self.portfolio):
            days_to_exp = p["entry_day"] + p["maturite"] - self.trading_day
            direction = "SELL" if p["is_short"] else "BUY "
            unrealized = (-1 if p["is_short"] else 1) * (
                p["last_price"] - p["entry_price"]
            ) * p["quantite"]
            color = GREEN if unrealized >= 0 else RED
            sign = "+" if unrealized >= 0 else ""

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            label = QLabel(
                "%s %4s K=%.1f T=%dd (%s%.2f)"
                % (direction, p["option_type"].upper(), p["strike"], days_to_exp, sign, unrealized)
            )
            label.setFont(QFont(MONO, 7))
            label.setStyleSheet(f"color: {color};")
            row_layout.addWidget(label, stretch=1)

            close = self._btn("CLOSE", RED, lambda _, idx=i: self._close_position(idx), outline=True)
            close.setFont(QFont(MONO, 7, QFont.Bold))
            row_layout.addWidget(close)
            self.port_layout.addWidget(row)

        self.port_layout.addStretch(1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TradingApp()
    window.show()
    sys.exit(app.exec_())

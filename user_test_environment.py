"""
Interactive trading desk on REAL Deribit BTC options data.

Loads day-by-day data from data/raw/ and lets the user trade.

Two P&L tracked:
    - realized P&L : actual cash movements (premiums paid/received, payoffs)
    - unrealized P&L : current market value of open positions
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore    import Qt
from PyQt5.QtGui     import QFont, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QScrollArea, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)

from data.loader import list_available_days, load_day


# ── Theme ─────────────────────────────────────────────────────────────────────

BG      = "#ffffff"
BG2     = "#f5f5f5"
BORDER  = "#dddddd"
GREEN   = "#00aa66"
RED     = "#dd2244"
FG      = "#111111"
FG2     = "#666666"
MONO    = "Courier"


class TradingApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL-4-Quant // Real Data Trading Desk")
        self.resize(1400, 880)

        # session state
        self.days           = list_available_days()
        self.day_index      = 0
        self.current_data   = None        # dict from load_day
        self.spot_history   = []
        self.portfolio      = []          # open positions
        self.realized_pnl   = 0.0
        self.unrealized_pnl = 0.0
        self.realized_hist  = []
        self.log            = []

        self._build_ui()

        if self.days:
            self._goto_day(0)

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        root.setStyleSheet(self._stylesheet())
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 8, 16, 16)
        root_layout.setSpacing(8)

        # top metric bar
        top = QHBoxLayout()
        top.setSpacing(10)
        title = QLabel("RL-4-Quant // Real Data")
        title.setFont(QFont(MONO, 15, QFont.Bold))
        top.addWidget(title, stretch=1)

        self.lbl_date     = self._metric(top, "DATE",       "—")
        self.lbl_spot     = self._metric(top, "BTC SPOT",   "—")
        self.lbl_options  = self._metric(top, "OPTIONS",    "—")
        self.lbl_realized = self._metric(top, "REALIZED P&L",   "0.00", color=GREEN)
        self.lbl_unreal   = self._metric(top, "UNREALIZED P&L", "0.00", color=GREEN)
        root_layout.addLayout(top)

        # main layout: left (chart + options) + right (controls)
        main = QHBoxLayout()
        main.setSpacing(12)
        root_layout.addLayout(main, stretch=1)

        # ── LEFT ──────────────────────────────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # spot chart
        self.fig = plt.Figure(figsize=(8, 3.5), facecolor=BG)
        gs = gridspec.GridSpec(1, 1, figure=self.fig)
        self.ax_spot = self.fig.add_subplot(gs[0])
        self._style_axes(self.ax_spot)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFixedHeight(220)
        left_layout.addWidget(self.canvas)

        # options table
        self._sep_above(left_layout, "OPTIONS OF THE DAY")
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(
            ["Type", "Strike", "Expiry", "Last (BTC)", "Avg (BTC)", "IV %", "Volume", "Qty", "Action"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        left_layout.addWidget(self.table, stretch=1)

        main.addWidget(left, stretch=1)

        # ── RIGHT (controls) ──────────────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(310)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self._build_controls(right_layout)
        main.addWidget(right)

    def _build_controls(self, parent):
        # navigation
        self._sep(parent, "NAVIGATE")
        nav = self._panel()
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(8, 8, 8, 8)
        nav_layout.addWidget(self._btn("▶  Next Day",  FG,   self._next_day))
        nav_layout.addWidget(self._btn("⏮  Prev Day",  FG2,  self._prev_day))
        nav_layout.addWidget(self._btn("↺  Restart",   FG2,  self._restart))
        parent.addWidget(nav)

        # portfolio
        self._sep(parent, "PORTFOLIO")
        self.port_frame = self._panel()
        self.port_layout = QVBoxLayout(self.port_frame)
        self.port_layout.setContentsMargins(4, 4, 4, 4)
        self.port_layout.setSpacing(2)

        port_scroll = QScrollArea()
        port_scroll.setWidgetResizable(True)
        port_scroll.setFixedHeight(200)
        port_scroll.setWidget(self.port_frame)
        parent.addWidget(port_scroll)

        # trade log
        self._sep(parent, "TRADE LOG")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont(MONO, 7))
        parent.addWidget(self.log_text, stretch=1)

    # ── Helpers UI ────────────────────────────────────────────────────────────

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
            padding: 2px 4px;
        }}
        QPushButton {{
            border: none;
            padding: 6px 8px;
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
        QTableWidget {{
            background: {BG};
            color: {FG};
            gridline-color: {BORDER};
            font-family: {MONO};
            font-size: 11px;
        }}
        QTableWidget::item {{
            padding: 2px 4px;
        }}
        QHeaderView::section {{
            background: {BG2};
            color: {FG2};
            border: none;
            padding: 4px;
            font-weight: bold;
            font-size: 10px;
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

    def _panel(self):
        panel = QFrame()
        panel.setStyleSheet(f"background: {BG2}; border: none;")
        return panel

    def _sep(self, parent, title):
        label = QLabel(title)
        label.setStyleSheet(f"color: {FG2};")
        label.setFont(QFont(MONO, 7))
        label.setContentsMargins(4, 10, 0, 1)
        parent.addWidget(label)

    def _sep_above(self, layout, title):
        label = QLabel(title)
        label.setStyleSheet(f"color: {FG2};")
        label.setFont(QFont(MONO, 8))
        layout.addWidget(label)

    def _btn(self, text, color, cmd, outline=False):
        button = QPushButton(text)
        button.setCursor(Qt.PointingHandCursor)
        button.setFont(QFont(MONO, 8, QFont.Bold))
        if outline:
            button.setStyleSheet(f"background: {BG2}; color: {color}; border: 1px solid {color};")
        else:
            button.setStyleSheet(f"background: {color}; color: {BG};")
        button.clicked.connect(cmd)
        return button

    def _style_axes(self, ax):
        ax.set_facecolor(BG2)
        ax.tick_params(colors=FG2, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _goto_day(self, idx: int):
        if idx < 0 or idx >= len(self.days):
            return
        self.day_index    = idx
        date_str          = self.days[idx]
        self.current_data = load_day(date_str)
        self.spot_history.append(self.current_data["spot"])
        self._mark_to_market()
        self.realized_hist.append(self.realized_pnl)
        self._refresh()

    def _next_day(self):
        if self.day_index + 1 < len(self.days):
            self._goto_day(self.day_index + 1)

    def _prev_day(self):
        if self.day_index > 0:
            # rewind one day: pop last spot/history (no rollback on portfolio)
            self.spot_history  = self.spot_history[:-1]
            self.realized_hist = self.realized_hist[:-1]
            self._goto_day(self.day_index - 1)

    def _restart(self):
        self.day_index      = 0
        self.spot_history   = []
        self.portfolio      = []
        self.realized_pnl   = 0.0
        self.unrealized_pnl = 0.0
        self.realized_hist  = []
        self.log            = []
        if self.days:
            self._goto_day(0)

    # ── Trades ────────────────────────────────────────────────────────────────

    def _order(self, opt: dict, qty: int, side: str):
        if qty <= 0:
            return
        price = opt["last_price"]
        if side == "BUY":
            self.realized_pnl -= price * qty           # pay the premium
        else:
            self.realized_pnl += price * qty           # receive the premium

        self.portfolio.append({
            **opt,
            "quantity":    qty,
            "is_short":    side == "SELL",
            "entry_price": price,
            "last_price":  price,
            "entry_date":  self.days[self.day_index],
        })
        self.log.append(
            f"{self.days[self.day_index]} | {side:4} {opt['option_type'].upper():4} "
            f"K={opt['strike']:.0f} {opt['expiry'].strftime('%d%b%y')} qty={qty} "
            f"@ {price:.4f} BTC"
        )
        self.realized_hist[-1] = self.realized_pnl
        self._refresh()

    def _close(self, idx: int):
        if idx >= len(self.portfolio):
            return
        p     = self.portfolio[idx]
        price = p["last_price"]
        # close = reverse trade at current market price
        if p["is_short"]:
            self.realized_pnl -= price * p["quantity"]    # buy back
        else:
            self.realized_pnl += price * p["quantity"]    # sell
        side = "BUY back" if p["is_short"] else "SELL"
        self.log.append(
            f"{self.days[self.day_index]} | CLOSE {side} {p['option_type'].upper():4} "
            f"K={p['strike']:.0f} qty={p['quantity']} @ {price:.4f} BTC"
        )
        self.portfolio.pop(idx)
        self.realized_hist[-1] = self.realized_pnl
        self._refresh()

    def _mark_to_market(self):
        """Update last_price of open positions using today's data, recompute unrealized P&L."""
        opts_by_instr = {o["instrument"]: o for o in self.current_data["options"]}
        unrealized = 0.0
        for p in self.portfolio:
            o = opts_by_instr.get(p["instrument"])
            if o is not None:
                p["last_price"] = o["last_price"]
            sign = -1 if p["is_short"] else 1
            unrealized += sign * (p["last_price"] - p["entry_price"]) * p["quantity"]
        self.unrealized_pnl = unrealized

    # ── Refresh UI ────────────────────────────────────────────────────────────

    def _refresh(self):
        d = self.current_data

        self.lbl_date.setText(d["date"])
        self.lbl_spot.setText(f"${d['spot']:.2f}")
        self.lbl_options.setText(str(len(d["options"])))
        self.lbl_realized.setText(f"{self.realized_pnl:+.4f} BTC")
        self.lbl_realized.setStyleSheet(
            f"color: {GREEN if self.realized_pnl >= 0 else RED};"
        )
        self.lbl_unreal.setText(f"{self.unrealized_pnl:+.4f} BTC")
        self.lbl_unreal.setStyleSheet(
            f"color: {GREEN if self.unrealized_pnl >= 0 else RED};"
        )

        self._draw_spot_chart()
        self._fill_options_table()
        self._refresh_portfolio()

        self.log_text.setPlainText("\n".join(reversed(self.log[-30:])))

    def _draw_spot_chart(self):
        self.ax_spot.cla()
        self._style_axes(self.ax_spot)
        self.ax_spot.plot(self.spot_history, color=GREEN, lw=1.5)
        self.ax_spot.set_title("BTC Spot (USD)", color=FG2, fontsize=8, pad=3)
        self.canvas.draw()

    def _fill_options_table(self):
        opts = sorted(self.current_data["options"], key=lambda o: (o["expiry"], o["strike"]))
        self.table.setRowCount(len(opts))

        for row, o in enumerate(opts):
            self.table.setItem(row, 0, QTableWidgetItem(o["option_type"].upper()))
            self.table.setItem(row, 1, QTableWidgetItem(f"{o['strike']:.0f}"))
            self.table.setItem(row, 2, QTableWidgetItem(o["expiry"].strftime("%d %b %y")))
            self.table.setItem(row, 3, QTableWidgetItem(f"{o['last_price']:.4f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{o['avg_price']:.4f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{o['iv']:.1f}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{o['volume']:.1f}"))

            # color call/put cell
            color = GREEN if o["option_type"] == "call" else RED
            self.table.item(row, 0).setForeground(QColor(color))

            # qty input
            qty_edit = QLineEdit("1")
            qty_edit.setFixedWidth(40)
            qty_edit.setAlignment(Qt.AlignRight)
            qty_edit.setFont(QFont(MONO, 9))
            self.table.setCellWidget(row, 7, qty_edit)

            # action buttons in last column
            action = QWidget()
            h = QHBoxLayout(action)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(2)
            buy  = QPushButton("BUY")
            sell = QPushButton("SELL")
            for b, col in [(buy, GREEN), (sell, RED)]:
                b.setFont(QFont(MONO, 7, QFont.Bold))
                b.setStyleSheet(f"background: {col}; color: {BG}; border: none; padding: 3px 6px;")
                b.setCursor(Qt.PointingHandCursor)
                h.addWidget(b)
            buy.clicked.connect (lambda _, o=o, e=qty_edit: self._order(o, self._safe_int(e), "BUY"))
            sell.clicked.connect(lambda _, o=o, e=qty_edit: self._order(o, self._safe_int(e), "SELL"))
            self.table.setCellWidget(row, 8, action)

    def _safe_int(self, edit):
        try:
            return int(edit.text())
        except ValueError:
            return 0

    def _refresh_portfolio(self):
        while self.port_layout.count():
            item = self.port_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not self.portfolio:
            empty = QLabel("No open positions.")
            empty.setStyleSheet(f"color: {FG2};")
            empty.setFont(QFont(MONO, 8))
            self.port_layout.addWidget(empty)
            self.port_layout.addStretch(1)
            return

        for i, p in enumerate(self.portfolio):
            side       = "SHORT" if p["is_short"] else "LONG "
            unrealized = (-1 if p["is_short"] else 1) * (p["last_price"] - p["entry_price"]) * p["quantity"]
            color      = GREEN if unrealized >= 0 else RED
            sign       = "+" if unrealized >= 0 else ""

            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)

            text = (f"{side} {p['option_type'].upper():4} K={p['strike']:.0f} "
                    f"qty={p['quantity']} ({sign}{unrealized:.4f})")
            label = QLabel(text)
            label.setFont(QFont(MONO, 7))
            label.setStyleSheet(f"color: {color};")
            h.addWidget(label, stretch=1)

            close = QPushButton("CLOSE")
            close.setFont(QFont(MONO, 7, QFont.Bold))
            close.setStyleSheet(f"background: {BG2}; color: {RED}; border: 1px solid {RED}; padding: 2px 6px;")
            close.setCursor(Qt.PointingHandCursor)
            close.clicked.connect(lambda _, idx=i: self._close(idx))
            h.addWidget(close)
            self.port_layout.addWidget(row)

        self.port_layout.addStretch(1)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TradingApp()
    window.show()
    sys.exit(app.exec_())
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec

from envs.underlying import Underlying
from envs.pricing import black_scholes

BG     = "#ffffff"
BG2    = "#f5f5f5"
BORDER = "#dddddd"
GREEN  = "#00aa66"
RED    = "#dd2244"
FG     = "#111111"
FG2    = "#666666"
MONO   = "Courier"


class TradingApp:

    def __init__(self, root):
        self.root = root
        self.root.title("RL-4-Quant // Trading Desk")
        self.root.configure(bg=BG)
        self.root.geometry("1200x820")
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
        top = tk.Frame(self.root, bg=BG, pady=8)
        top.pack(fill="x", padx=16)
        tk.Label(top, text="RL-4-Quant // Trading Desk", bg=BG, fg=FG,
                 font=(MONO, 15, "bold")).pack(side="left")
        self.lbl_day  = self._metric(top, "DAY",      "0")
        self.lbl_spot = self._metric(top, "SPOT",     "-")
        self.lbl_vol  = self._metric(top, "REAL VOL", "-")
        self.lbl_pnl  = self._metric(top, "P&L",      "0.00", color=GREEN)

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=4)

        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        self.fig = plt.Figure(figsize=(7, 5.5), facecolor=BG)
        gs = gridspec.GridSpec(2, 1, figure=self.fig, hspace=0.45, height_ratios=[2, 1])
        self.ax_price = self.fig.add_subplot(gs[0])
        self.ax_pnl   = self.fig.add_subplot(gs[1])
        for ax in [self.ax_price, self.ax_pnl]:
            ax.set_facecolor(BG2)
            ax.tick_params(colors=FG2, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        right = tk.Frame(main, bg=BG, width=290)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        self._build_controls(right)

    def _metric(self, parent, label, value, color=FG):
        f = tk.Frame(parent, bg=BG2, padx=12, pady=6)
        f.pack(side="right", padx=5)
        tk.Label(f, text=label, bg=BG2, fg=FG2, font=(MONO, 7)).pack()
        lbl = tk.Label(f, text=value, bg=BG2, fg=color, font=(MONO, 12, "bold"))
        lbl.pack()
        return lbl

    def _build_controls(self, parent):
        self._sep(parent, "CONFIG")
        cfg = tk.Frame(parent, bg=BG2, padx=8, pady=8)
        cfg.pack(fill="x")
        self.s0_var  = self._entry_row(cfg, "S0",       100.0)
        self.vol_var = self._entry_row(cfg, "Daily Vol", 0.01)
        self._btn(cfg, "New Episode", FG, self._new_episode).pack(fill="x", pady=(6, 0))

        self._sep(parent, "PLACE TRADE")
        trade = tk.Frame(parent, bg=BG2, padx=8, pady=8)
        trade.pack(fill="x")
        self.strike_var = self._entry_row(trade, "Strike K",    100.0)
        self.mat_var    = self._entry_row(trade, "Maturity (d)", 30,  is_int=True)
        self.qty_var    = self._entry_row(trade, "Quantity",      1,  is_int=True)

        pf = tk.Frame(trade, bg=BG2, pady=4)
        pf.pack(fill="x")
        tk.Label(pf, text="CALL", bg=BG2, fg=FG2, font=(MONO, 8), width=6, anchor="w").grid(row=0, column=0)
        self.lbl_call = tk.Label(pf, text="-", bg=BG2, fg=GREEN, font=(MONO, 10, "bold"))
        self.lbl_call.grid(row=0, column=1, sticky="e")
        tk.Label(pf, text="PUT",  bg=BG2, fg=FG2, font=(MONO, 8), width=6, anchor="w").grid(row=1, column=0)
        self.lbl_put = tk.Label(pf, text="-", bg=BG2, fg=RED, font=(MONO, 10, "bold"))
        self.lbl_put.grid(row=1, column=1, sticky="e")
        pf.columnconfigure(1, weight=1)

        for var in (self.strike_var, self.mat_var):
            var.trace_add("write", lambda *_: self._update_prices())

        bf = tk.Frame(trade, bg=BG2)
        bf.pack(fill="x", pady=(6, 0))
        self._btn(bf, "BUY CALL",  GREEN, lambda: self._order("call", False)).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self._btn(bf, "BUY PUT",   RED,   lambda: self._order("put",  False)).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        self._btn(bf, "SELL CALL", GREEN, lambda: self._order("call", True),  outline=True).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        self._btn(bf, "SELL PUT",  RED,   lambda: self._order("put",  True),  outline=True).grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        bf.columnconfigure(0, weight=1)
        bf.columnconfigure(1, weight=1)

        self._sep(parent, "NAVIGATE")
        nav = tk.Frame(parent, bg=BG2, padx=8, pady=8)
        nav.pack(fill="x")
        self._btn(nav, "Next Day",  FG,  self._next_day).pack(fill="x", pady=2)
        self._btn(nav, "Skip Week", FG2, self._skip_week).pack(fill="x", pady=2)

        self._sep(parent, "PORTFOLIO")
        self.port_frame = tk.Frame(parent, bg=BG2)
        self.port_frame.pack(fill="x")

        self._sep(parent, "TRADE LOG")
        log_f = tk.Frame(parent, bg=BG2)
        log_f.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_f, bg=BG2, fg=FG2, font=(MONO, 7),
                                height=5, relief="flat", state="disabled")
        self.log_text.pack(fill="both", padx=4, pady=4)

    def _sep(self, parent, title):
        tk.Label(parent, text=title, bg=BG, fg=FG2,
                 font=(MONO, 7), anchor="w").pack(fill="x", pady=(10, 1), padx=4)

    def _entry_row(self, parent, label, default, is_int=False):
        row = tk.Frame(parent, bg=BG2)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg=BG2, fg=FG2, font=(MONO, 8),
                 width=13, anchor="w").pack(side="left")
        var = tk.IntVar(value=int(default)) if is_int else tk.DoubleVar(value=default)
        tk.Entry(row, textvariable=var, bg=BG, fg=FG, insertbackground=FG,
                 relief="flat", font=(MONO, 9), width=9).pack(side="right")
        return var

    def _btn(self, parent, text, color, cmd, outline=False):
        bg = BG2 if outline else color
        fg = color if outline else BG
        return tk.Button(parent, text=text, bg=bg, fg=fg, relief="flat",
                         font=(MONO, 8, "bold"), cursor="hand2",
                         activebackground=color, activeforeground=BG,
                         command=cmd)

    def _new_episode(self):
        try:
            self.S0        = float(self.s0_var.get())
            self.daily_vol = float(self.vol_var.get())
        except Exception:
            pass
        self.underlying = Underlying(S0=self.S0, daily_vol=self.daily_vol)
        self.underlying.simulate(251)
        self.history_prices = list(self.underlying.prices)
        self.live_prices = [self.history_prices[-1]]
        self.trading_day = 0
        self.portfolio   = []
        self.pnl_history = [0.0]
        self.total_pnl   = 0.0
        self.log         = []
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
            K   = float(self.strike_var.get())
            mat = int(self.mat_var.get())
            qty = int(self.qty_var.get())
        except Exception:
            return
        if qty <= 0 or mat <= 0:
            return
        S     = self._current_price()
        sigma = self._realized_vol()
        T     = mat / 252
        price = black_scholes(S=S, K=K, T=T, r=self.r, sigma=sigma, option_type=option_type)
        direction = "SELL" if is_short else "BUY "
        self.portfolio.append({
            "option_type": option_type,
            "strike":      K,
            "maturite":    mat,
            "quantite":    qty,
            "is_short":    is_short,
            "entry_day":   self.trading_day,
            "entry_price": price,
            "last_price":  price,
        })
        # broker fees: 0.65 per contract
        fees = 0.65 * qty
        self.total_pnl -= fees
        self.pnl_history[-1] = self.total_pnl

        self.log.append(
            "Day %3d | %s %4s K=%.1f T=%dd qty=%d @ %.4f  fees=-%.2f" % (
                self.trading_day, direction, option_type.upper(), K, mat, qty, price, fees)
        )
        self._refresh()

    def _close_position(self, idx):
        p     = self.portfolio[idx]
        price = p["last_price"]
        direction = -1 if p["is_short"] else 1
        # realize the unrealized P&L immediately
        realized = direction * (price - p["entry_price"]) * p["quantite"]
        # broker fees on close too
        fees = 0.65 * p["quantite"]
        self.total_pnl -= fees
        self.pnl_history[-1] = self.total_pnl

        self.log.append(
            "Day %3d | CLOSE %4s K=%.1f @ %.4f  P&L=%.2f  fees=-%.2f" % (
                self.trading_day, p["option_type"].upper(), p["strike"], price, realized, fees)
        )
        self.portfolio.pop(idx)
        self._refresh()

    def _next_day(self):
        self._advance()
        self._refresh()

    def _skip_week(self):
        for _ in range(5):
            self._advance()
        self._refresh()

    def _advance(self):
        S     = self.live_prices[-1]
        Z     = np.random.standard_normal()
        S_new = S * np.exp(-0.5 * self.daily_vol ** 2 + self.daily_vol * Z)
        self.live_prices.append(S_new)
        self.trading_day += 1
        self._check_early_exercise()
        self.total_pnl   += self._mark_to_market()
        self.pnl_history.append(self.total_pnl)

    def _check_early_exercise(self):
        """For short positions: if ITM, the buyer exercises immediately."""
        S = self._current_price()
        for p in self.portfolio:
            if not p["is_short"]:
                continue
            if p["option_type"] == "call" and S > p["strike"]:
                # call exerced: we lose (S - K) * qty, minus what we already lost via mtm
                payoff = (S - p["strike"]) * p["quantite"]
                loss   = -payoff - (-p["last_price"] * p["quantite"])  # net vs last mtm
                self.total_pnl += loss
                self.log.append(
                    f"Day {self.trading_day:>3} | EXERCISED CALL K={p['strike']:.1f} S={S:.2f}  loss={(loss):+.4f}"
                )
                p["expired"] = True
            elif p["option_type"] == "put" and S < p["strike"]:
                # put exerced: we lose (K - S) * qty
                payoff = (p["strike"] - S) * p["quantite"]
                loss   = -payoff - (-p["last_price"] * p["quantite"])
                self.total_pnl += loss
                self.log.append(
                    f"Day {self.trading_day:>3} | EXERCISED PUT  K={p['strike']:.1f} S={S:.2f}  loss={(loss):+.4f}"
                )
                p["expired"] = True
        self.portfolio = [p for p in self.portfolio if not p.get("expired")]

    def _mark_to_market(self):
        S     = self._current_price()
        sigma = self._realized_vol()
        t     = self.trading_day
        daily_pnl = 0.0
        for p in self.portfolio:
            days_left = p["entry_day"] + p["maturite"] - t
            if days_left <= 0:
                payoff = max(S - p["strike"], 0.0) if p["option_type"] == "call" else max(p["strike"] - S, 0.0)
                direction = -1 if p["is_short"] else 1
                daily_pnl += direction * (payoff - p["last_price"]) * p["quantite"]
                p["last_price"] = payoff
                p["expired"]    = True
            else:
                T         = days_left / 252
                new_price = black_scholes(S=S, K=p["strike"], T=T, r=self.r,
                                          sigma=sigma, option_type=p["option_type"])
                direction = -1 if p["is_short"] else 1
                daily_pnl += direction * (new_price - p["last_price"]) * p["quantite"]
                p["last_price"] = new_price
        self.portfolio = [p for p in self.portfolio if not p.get("expired")]
        return daily_pnl

    def _close_position(self, idx):
        if idx >= len(self.portfolio):
            return
        p         = self.portfolio[idx]
        direction = "SELL" if p["is_short"] else "BUY "
        # realize P&L at current market price (last_price already up to date)
        realized  = (-1 if p["is_short"] else 1) * (p["last_price"] - p["entry_price"]) * p["quantite"]
        self.log.append(
            f"Day {self.trading_day:>3} | CLOSE {direction.strip()} {p['option_type'].upper():4} "
            f"K={p['strike']:.1f} @ {p['last_price']:.4f}  PnL={realized:+.4f}"
        )
        self.portfolio.pop(idx)
        self._refresh()

    def _update_prices(self):
        if not self.live_prices:
            return
        try:
            K   = float(self.strike_var.get())
            mat = max(int(self.mat_var.get()), 1)
        except Exception:
            return
        S     = self._current_price()
        sigma = self._realized_vol()
        T     = mat / 252
        call_p = black_scholes(S=S, K=K, T=T, r=self.r, sigma=sigma, option_type="call")
        put_p  = black_scholes(S=S, K=K, T=T, r=self.r, sigma=sigma, option_type="put")
        self.lbl_call.config(text="%.4f" % call_p)
        self.lbl_put.config(text="%.4f" % put_p)

    def _refresh(self):
        S     = self._current_price()
        sigma = self._realized_vol()
        pnl   = self.total_pnl

        self.lbl_day.config(text=str(self.trading_day))
        self.lbl_spot.config(text="%.2f" % S)
        self.lbl_vol.config(text="%.1f%%" % (sigma * 100))
        self.lbl_pnl.config(text="%+.2f" % pnl, fg=GREEN if pnl >= 0 else RED)

        all_p  = self._all_prices()
        n_hist = len(self.history_prices)
        n_total = len(all_p)

        self.ax_price.cla()
        self.ax_pnl.cla()
        for ax in [self.ax_price, self.ax_pnl]:
            ax.set_facecolor(BG2)
            ax.tick_params(colors=FG2, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)

        self.ax_price.plot(range(n_hist), self.history_prices, color=BORDER, lw=1.5)
        if len(self.live_prices) > 1:
            live_x = list(range(n_hist - 1, n_total))
            self.ax_price.plot(live_x, [self.history_prices[-1]] + self.live_prices[1:],
                               color=GREEN, lw=1.5)
        self.ax_price.axvline(x=n_hist - 1, color=BORDER, lw=1, ls="--")
        self.ax_price.set_title("grey = history   green = trading", color=FG2, fontsize=8, pad=3)

        pnl_hist   = self.pnl_history
        fill_color = GREEN if pnl_hist[-1] >= 0 else RED
        self.ax_pnl.plot(pnl_hist, color=fill_color, lw=1.5)
        self.ax_pnl.fill_between(range(len(pnl_hist)), pnl_hist, alpha=0.15, color=fill_color)
        self.ax_pnl.axhline(0, color=BORDER, lw=0.8)
        self.ax_pnl.set_title("P&L", color=FG2, fontsize=8, pad=3)

        self.canvas.draw()

        # portfolio: one row per position with a CLOSE button
        for w in self.port_frame.winfo_children():
            w.destroy()
        if not self.portfolio:
            tk.Label(self.port_frame, text="No open positions.", bg=BG2, fg=FG2,
                     font=(MONO, 8)).pack(anchor="w", padx=4, pady=4)
        else:
            for i, p in enumerate(self.portfolio):
                days_to_exp = p["entry_day"] + p["maturite"] - self.trading_day
                direction   = "SELL" if p["is_short"] else "BUY "
                unrealized  = (-1 if p["is_short"] else 1) * (p["last_price"] - p["entry_price"]) * p["quantite"]
                color       = GREEN if unrealized >= 0 else RED
                sign        = "+" if unrealized >= 0 else ""
                row = tk.Frame(self.port_frame, bg=BG2)
                row.pack(fill="x", padx=4, pady=1)
                tk.Label(row,
                         text="%s %4s K=%.1f T=%dd (%s%.2f)" % (
                             direction, p["option_type"].upper(), p["strike"], days_to_exp, sign, unrealized),
                         bg=BG2, fg=color, font=(MONO, 7), anchor="w").pack(side="left")
                tk.Button(row, text="CLOSE", bg=BG2, fg=RED, font=(MONO, 7, "bold"),
                          relief="flat", cursor="hand2",
                          command=lambda idx=i: self._close_position(idx)).pack(side="right")

        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        for entry in reversed(self.log[-30:]):
            self.log_text.insert("end", entry + "\n")
        self.log_text.config(state="disabled")

        self._update_prices()


if __name__ == "__main__":
    root = tk.Tk()
    TradingApp(root)
    root.mainloop()
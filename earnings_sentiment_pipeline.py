import os, time, warnings, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.facecolor": "#0d1117",
    "figure.facecolor": "#0d1117", "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3", "xtick.color": "#e6edf3",
    "ytick.color": "#e6edf3", "axes.edgecolor": "#30363d", "grid.color": "#21262d",
})
ACCENT_GREEN = "#39d353"; ACCENT_RED = "#f85149"; ACCENT_BLUE = "#58a6ff"
ACCENT_GOLD = "#e3b341"; ACCENT_PURPLE = "#bc8cff"; ACCENT_TEAL = "#39d3c3"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


class EarningsDataEngine:
    SECTORS = {
        "Technology":  ["AAPL","MSFT","GOOGL","META","NVDA","AMD","INTC","CRM"],
        "Finance":     ["JPM","BAC","GS","MS","WFC","C","BLK","AXP"],
        "Healthcare":  ["JNJ","PFE","UNH","ABBV","MRK","TMO","ABT","BMY"],
        "Consumer":    ["AMZN","TSLA","NKE","SBUX","MCD","TGT","WMT","HD"],
        "Energy":      ["XOM","CVX","COP","SLB","EOG","PXD","MPC","PSX"],
        "Industrials": ["BA","CAT","GE","MMM","HON","UPS","FDX","RTX"],
    }
    HEADLINES = {
        "strong_beat": [
            "{t} crushes Q{q} estimates; EPS beats by {p:.1f}%, raises full-year outlook",
            "{t} posts record revenue in Q{q}, EPS surprises by {p:.1f}%",
            "{t} delivers blockbuster Q{q}: earnings surge {p:.1f}% above Wall Street",
            "{t} Q{q} blowout: revenue and margins exceed consensus, guidance raised",
        ],
        "moderate_beat": [
            "{t} Q{q} earnings top estimates by {p:.1f}%; guidance maintained",
            "{t} reports Q{q} beat: EPS ahead by {p:.1f}%, in-line revenue",
            "{t} Q{q}: modest beat on EPS, revenue slightly above expectations",
            "{t} edges past Q{q} consensus; management reaffirms targets",
        ],
        "in_line": [
            "{t} Q{q} results in line with analyst expectations; outlook unchanged",
            "{t} meets Q{q} estimates: EPS at consensus, revenue matches forecast",
            "{t} Q{q} earnings steady; no major surprises as guidance maintained",
            "{t} delivers Q{q} results matching Street estimates",
        ],
        "moderate_miss": [
            "{t} Q{q} EPS misses by {p:.1f}%; margin pressure cited amid rising costs",
            "{t} falls short of Q{q} estimates; revenue disappoints, guidance trimmed",
            "{t} Q{q} miss: earnings below consensus by {p:.1f}%, shares slide",
            "{t} reports below-consensus Q{q}; macro headwinds weigh on results",
        ],
        "strong_miss": [
            "{t} Q{q} disaster: EPS misses by {p:.1f}%, full-year guidance slashed",
            "{t} posts steep Q{q} shortfall; revenue collapses {p:.1f}% below forecast",
            "{t} issues profit warning after Q{q} miss; CEO cites demand shock",
            "{t} Q{q} debacle: EPS and revenue miss, dividend cut under consideration",
        ],
    }

    @staticmethod
    def generate(n=2000, seed=42):
        np.random.seed(seed); rng = np.random.default_rng(seed)
        engine = EarningsDataEngine()
        ts = {t: s for s, tl in engine.SECTORS.items() for t in tl}
        tickers = rng.choice(list(ts.keys()), size=n, replace=True)
        start = datetime(2019, 1, 1)
        d_range = (datetime(2024, 6, 30) - start).days
        dates = [start + timedelta(days=int(d)) for d in rng.integers(0, d_range, n)]
        quarters = [f"Q{((d.month-1)//3)+1}" for d in dates]
        sectors = [ts[t] for t in tickers]
        eb = {"Technology":3.5,"Finance":4.2,"Healthcare":2.8,"Consumer":2.1,"Energy":5.1,"Industrials":3.0}
        rb = {"Technology":85,"Finance":120,"Healthcare":55,"Consumer":95,"Energy":140,"Industrials":65}
        eps_b = np.array([eb[s] for s in sectors]); rev_b = np.array([rb[s] for s in sectors])
        ab = rng.normal(0.02, 0.15, n)
        eps_est = eps_b * (1 + ab + rng.normal(0, 0.08, n))
        rev_est = rev_b * (1 + ab + rng.normal(0, 0.05, n))
        sk = rng.choice([-1, 0, 1], p=[0.25, 0.35, 0.40], size=n)
        esp = rng.normal(sk * 0.04, 0.08, n); rsp = rng.normal(sk * 0.02, 0.04, n)
        eps_act = eps_est * (1 + esp); rev_act = rev_est * (1 + rsp)
        gp = np.clip(0.45 + esp * 2, 0.05, 0.90)
        guidance = np.where(rng.random(n) < gp,
                            rng.choice(["Raised","Maintained"], p=[0.6,0.4], size=n), "Lowered")
        pre = rng.normal(esp * 0.3, 0.015, n)
        gf = np.where(guidance=="Raised", 0.012, np.where(guidance=="Lowered", -0.015, 0.0))
        dor = esp*1.8 + rsp*0.6 + gf + rng.normal(0, 0.025, n)
        d3  = dor*0.3 + rng.normal(0, 0.018, n); d10 = dor*0.2 + rng.normal(0, 0.03, n)
        d30 = dor*0.1 + rng.normal(0, 0.05, n); mkt = rng.normal(0.0003, 0.01, n)
        ar1 = dor - mkt; ar10 = (d3 + d10) - mkt * 10
        def cls(p):
            if p > 0.10: return "Strong Beat"
            elif p > 0.02: return "Moderate Beat"
            elif p > -0.02: return "In-Line"
            elif p > -0.10: return "Moderate Miss"
            else: return "Strong Miss"
        km = {"Strong Beat":"strong_beat","Moderate Beat":"moderate_beat",
              "In-Line":"in_line","Moderate Miss":"moderate_miss","Strong Miss":"strong_miss"}
        scs = [cls(p) for p in esp]; heads = []
        for i, (tk, q, sc, ep) in enumerate(zip(tickers, quarters, scs, esp)):
            tmps = engine.HEADLINES[km[sc]]
            heads.append(tmps[rng.integers(0, len(tmps))].format(t=tk, q=q[1], p=abs(ep)*100))
        df = pd.DataFrame({
            "ticker": tickers, "sector": sectors, "earnings_date": dates,
            "fiscal_quarter": quarters, "fiscal_year": [d.year for d in dates],
            "eps_estimate": eps_est.round(2), "eps_actual": eps_act.round(2),
            "eps_surprise_pct": (esp * 100).round(2),
            "rev_estimate_bn": rev_est.round(1), "rev_actual_bn": rev_act.round(1),
            "rev_surprise_pct": (rsp * 100).round(2),
            "guidance": guidance, "surprise_class": scs,
            "pre_earnings_drift": (pre * 100).round(2),
            "day_of_reaction_pct": (dor * 100).round(2),
            "drift_3d_pct": (d3 * 100).round(2), "drift_10d_pct": (d10 * 100).round(2),
            "drift_30d_pct": (d30 * 100).round(2), "abnormal_return_1d": (ar1 * 100).round(2),
            "abnormal_return_10d": (ar10 * 100).round(2), "headline": heads,
        })
        return df.sort_values("earnings_date").reset_index(drop=True)


class FinancialSentimentEngine:
    POS = {"record":2.5,"blowout":2.8,"blockbuster":2.6,"surge":2.0,"soars":2.2,
           "crushes":2.4,"beats":1.8,"raised":1.9,"tops":1.6,"beat":1.7,"ahead":1.3,
           "above":1.2,"exceeds":1.7,"reaffirms":1.1,"steady":0.8,"strong":1.4,
           "growth":1.2,"robust":1.5,"outperform":1.8,"upgrade":1.6,"bullish":1.7,
           "positive":1.0,"improvement":1.1,"success":1.3,"exceed":1.5,"upside":1.4}
    NEG = {"disaster":-2.8,"slashed":-2.5,"collapses":-2.6,"debacle":-2.7,
           "warning":-1.8,"cut":-1.4,"misses":-1.9,"disappoints":-1.7,"below":-1.2,
           "falls":-1.3,"slides":-1.5,"trimmed":-1.3,"miss":-1.6,"shortfall":-2.0,
           "pressure":-1.2,"headwinds":-1.4,"decline":-1.3,"drop":-1.4,"loss":-1.8,
           "weak":-1.2,"disappointing":-1.7,"concern":-1.0,"risk":-0.8,"shock":-1.5,
           "negative":-1.0,"uncertainty":-0.9,"downgrade":-1.7}
    NEG_WORDS = {"not","no","never","neither","without","barely","hardly"}
    INT_WORDS = {"very","highly","extremely","significantly","sharply","substantially"}

    def score(self, text):
        words = re.findall(r"\b\w+\b", text.lower())
        sc, ws = 0.0, []
        for i, w in enumerate(words):
            base = self.POS.get(w, 0.0) + self.NEG.get(w, 0.0)
            if base == 0.0: continue
            if any(x in self.NEG_WORDS for x in words[max(0, i-2):i]): base *= -0.75
            if i > 0 and words[i-1] in self.INT_WORDS: base *= 1.4
            sc += base; ws.append(base)
        n = len(ws) if ws else 1; norm = float(np.tanh(sc / (n * 1.5)))
        if norm > 0.5:      lbl = "Very Positive"
        elif norm > 0.15:   lbl = "Positive"
        elif norm > -0.15:  lbl = "Neutral"
        elif norm > -0.5:   lbl = "Negative"
        else:               lbl = "Very Negative"
        return {"sentiment_score": round(norm,4), "sentiment_label": lbl,
                "confidence": round(min(n/5.0,1.0),3), "n_words": n}

    def score_df(self, df):
        print("  [NLP] Scoring all earnings headlines...")
        res = [self.score(h) for h in df["headline"]]; df = df.copy()
        df["sentiment_score"]   = [r["sentiment_score"] for r in res]
        df["sentiment_label"]   = [r["sentiment_label"] for r in res]
        df["nlp_confidence"]    = [r["confidence"] for r in res]
        df["n_sentiment_words"] = [r["n_words"] for r in res]
        avg_s = df["sentiment_score"].mean()
        print(f"  [NLP] Avg sentiment = {avg_s:.3f}")
        return df


class EarningsSurpriseAnalytics:
    def compute(self, df):
        df = df.copy()
        print("  [Analytics] Computing EQS and composite signals...")
        se = df.groupby("sector")["eps_surprise_pct"].transform("std").fillna(1.0)
        sr = df.groupby("sector")["rev_surprise_pct"].transform("std").fillna(1.0)
        df["eps_z_score"] = (df["eps_surprise_pct"] / se).clip(-4, 4)
        df["rev_z_score"] = (df["rev_surprise_pct"] / sr).clip(-4, 4)
        df["guidance_score"] = df["guidance"].map({"Raised":1.0,"Maintained":0.0,"Lowered":-1.0})
        df["earnings_quality_score"] = (0.45*df["eps_z_score"] + 0.30*df["rev_z_score"]
                                        + 0.25*df["guidance_score"]*2).round(3)
        df["composite_signal"] = (0.60*df["earnings_quality_score"]
                                  + 0.40*df["sentiment_score"]*3).round(3)
        df["signal"] = pd.cut(df["composite_signal"],
                              bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
                              labels=["Strong Sell","Sell","Hold","Buy","Strong Buy"])
        df["positive_outcome"] = (df["abnormal_return_1d"] > 0).astype(int)
        return df

    def sector_stats(self, df):
        return df.groupby("sector").agg(
            n=("ticker","count"),
            avg_eps_surprise=("eps_surprise_pct","mean"),
            beat_rate=("surprise_class",
                lambda x: x.isin(["Strong Beat","Moderate Beat"]).mean()),
            avg_reaction=("day_of_reaction_pct","mean"),
            avg_sentiment=("sentiment_score","mean"),
            avg_eqs=("earnings_quality_score","mean"),
            guidance_raise_rate=("guidance", lambda x: (x=="Raised").mean()),
        ).round(3).reset_index()

    def ticker_stats(self, df):
        return df.groupby(["ticker","sector"]).agg(
            n=("ticker","count"),
            avg_eps_surprise=("eps_surprise_pct","mean"),
            avg_reaction=("day_of_reaction_pct","mean"),
            avg_eqs=("earnings_quality_score","mean"),
        ).round(3).reset_index().sort_values("avg_eqs", ascending=False)


class MLEngine:
    FEATS = ["eps_z_score","rev_z_score","guidance_score","sentiment_score",
             "nlp_confidence","n_sentiment_words","pre_earnings_drift"]

    def __init__(self): self.models = {}; self.le = LabelEncoder()

    def train(self, df):
        print("\n  [ML] Running model tournament...")
        d = df.copy().dropna(subset=self.FEATS + ["positive_outcome"])
        d["sec_enc"] = self.le.fit_transform(d["sector"])
        feats = self.FEATS + ["sec_enc"]
        X, y = d[feats].values, d["positive_outcome"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        cands = {
            "Logistic Regression": Pipeline([("sc",StandardScaler()),
                ("clf",LogisticRegression(C=0.5,max_iter=500,random_state=42))]),
            "Random Forest":       Pipeline([("sc",StandardScaler()),
                ("clf",RandomForestClassifier(n_estimators=200,max_depth=6,
                                              min_samples_leaf=20,random_state=42))]),
            "Gradient Boosting":   Pipeline([("sc",StandardScaler()),
                ("clf",GradientBoostingClassifier(n_estimators=150,max_depth=4,
                                                  learning_rate=0.05,random_state=42))]),
        }
        results = {}
        for name, pipe in cands.items():
            cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="roc_auc").mean()
            pipe.fit(Xtr, ytr); yp = pipe.predict_proba(Xte)[:,1]; ta = roc_auc_score(yte, yp)
            results[name] = {"model":pipe,"cv_auc":cv,"test_auc":ta,
                             "y_test":yte,"y_prob":yp,"y_pred":pipe.predict(Xte)}
            self.models[name] = pipe
            print(f"    {name:<25} CV:{cv:.4f}  Test:{ta:.4f}")
        best = max(results, key=lambda k: results[k]["test_auc"])
        best_auc = results[best]["test_auc"]
        print(f"\n  [ML] Best: {best} (AUC={best_auc:.4f})")
        return results, feats

    def feature_importance(self, feats):
        if "Random Forest" not in self.models: return pd.DataFrame()
        clf = self.models["Random Forest"].named_steps["clf"]
        return pd.DataFrame({"feature":feats,"importance":clf.feature_importances_}
                           ).sort_values("importance", ascending=False)


class VizEngine:
    def __init__(self, out): self.out = out; self.saved = []

    def _save(self, fig, name):
        p = os.path.join(self.out, name)
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig); self.saved.append(name)
        print(f"  [Chart] {name}")

    def plot_surprise_dist(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Earnings Surprise Distribution", fontsize=18, fontweight="bold", color="white", y=1.02)
        cm = {"Strong Beat":ACCENT_GREEN,"Moderate Beat":"#88d8a3","In-Line":ACCENT_GOLD,
              "Moderate Miss":"#f4a37d","Strong Miss":ACCENT_RED}
        ax = axes[0]
        for sc, g in df.groupby("surprise_class"):
            ax.hist(g["eps_surprise_pct"], bins=30, alpha=0.75, color=cm.get(sc,ACCENT_BLUE), label=sc, edgecolor="none")
        ax.axvline(0, color="white", lw=1.5, ls="--", alpha=0.7)
        ax.set_title("EPS Surprise % by Class", color="white", fontsize=13)
        ax.set_xlabel("EPS Surprise (%)", color="#8b949e"); ax.set_ylabel("Frequency", color="#8b949e")
        ax.legend(fontsize=9, framealpha=0.3); ax.tick_params(colors="#8b949e")
        ax2 = axes[1]
        br = df.groupby("sector").apply(
            lambda x: x["surprise_class"].isin(["Strong Beat","Moderate Beat"]).mean()*100).sort_values()
        c2 = [ACCENT_GREEN if v>55 else ACCENT_GOLD if v>45 else ACCENT_RED for v in br.values]
        bars = ax2.barh(br.index, br.values, color=c2, edgecolor="none", height=0.6)
        ax2.axvline(50, color="white", lw=1, ls="--", alpha=0.5)
        ax2.set_title("Beat Rate by Sector (%)", color="white", fontsize=13)
        ax2.set_xlabel("% Beating Consensus", color="#8b949e"); ax2.tick_params(colors="#8b949e")
        for bar, v in zip(bars, br.values):
            ax2.text(v+0.5, bar.get_y()+bar.get_height()/2, f"{v:.1f}%", va="center", fontsize=9, color="white")
        fig.tight_layout(); self._save(fig, "chart_01_surprise_distribution.png")

    def plot_reaction_scatter(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Earnings Surprise vs Market Reaction", fontsize=18, fontweight="bold", color="white", y=1.02)
        cm = {"Strong Beat":ACCENT_GREEN,"Moderate Beat":"#88d8a3","In-Line":ACCENT_GOLD,
              "Moderate Miss":"#f4a37d","Strong Miss":ACCENT_RED}
        ax = axes[0]; smp = df.sample(min(1000, len(df)), random_state=42)
        for sc, g in smp.groupby("surprise_class"):
            ax.scatter(g["eps_surprise_pct"], g["day_of_reaction_pct"], c=cm.get(sc,"white"), alpha=0.6, s=30, label=sc)
        z = np.polyfit(smp["eps_surprise_pct"], smp["day_of_reaction_pct"], 1)
        xl = np.linspace(smp["eps_surprise_pct"].min(), smp["eps_surprise_pct"].max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), color=ACCENT_BLUE, lw=2, ls="--", alpha=0.8, label="Trend")
        ax.axhline(0, color="white", lw=0.8, alpha=0.4); ax.axvline(0, color="white", lw=0.8, alpha=0.4)
        ax.set_title("EPS Surprise vs Day-of Reaction", color="white", fontsize=13)
        ax.set_xlabel("EPS Surprise (%)", color="#8b949e"); ax.set_ylabel("Day-of Reaction (%)", color="#8b949e")
        ax.legend(fontsize=8, framealpha=0.3); ax.tick_params(colors="#8b949e")
        ax2 = axes[1]
        order = ["Strong Miss","Moderate Miss","In-Line","Moderate Beat","Strong Beat"]
        data = [df[df["surprise_class"]==sc]["day_of_reaction_pct"].values for sc in order]
        bp = ax2.boxplot(data, patch_artist=True, notch=True, medianprops=dict(color="white", lw=2))
        for patch, c in zip(bp["boxes"], [ACCENT_RED,"#f4a37d",ACCENT_GOLD,"#88d8a3",ACCENT_GREEN]):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax2.set_xticklabels([s.replace(" ","\n") for s in order], color="#8b949e", fontsize=8)
        ax2.axhline(0, color="white", lw=0.8, alpha=0.4)
        ax2.set_title("Reaction by Surprise Class", color="white", fontsize=13)
        ax2.set_ylabel("Day-of Reaction (%)", color="#8b949e"); ax2.tick_params(colors="#8b949e")
        fig.tight_layout(); self._save(fig, "chart_02_surprise_reaction.png")

    def plot_sentiment(self, df):
        fig, axes = plt.subplots(2, 2, figsize=(18, 13))
        fig.suptitle("Earnings Sentiment NLP Analysis", fontsize=18, fontweight="bold", color="white")
        sc_c = {"Very Positive":ACCENT_GREEN,"Positive":"#88d8a3","Neutral":ACCENT_GOLD,
                "Negative":"#f4a37d","Very Negative":ACCENT_RED}
        lo = ["Very Negative","Negative","Neutral","Positive","Very Positive"]
        ax = axes[0][0]; counts = df["sentiment_label"].value_counts().reindex(lo, fill_value=0)
        bars = ax.bar(counts.index, counts.values, color=[sc_c[l] for l in counts.index], edgecolor="none")
        ax.set_title("Headline Sentiment Distribution", color="white", fontsize=13)
        ax.set_xlabel("Sentiment", color="#8b949e"); ax.set_ylabel("Count", color="#8b949e")
        ax.tick_params(axis="x", rotation=30, colors="#8b949e"); ax.tick_params(axis="y", colors="#8b949e")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    f"{int(bar.get_height()):,}", ha="center", fontsize=9, color="white")
        ax2 = axes[0][1]
        s2 = ax2.scatter(df["sentiment_score"], df["day_of_reaction_pct"],
                         c=df["day_of_reaction_pct"], cmap="RdYlGn", alpha=0.4, s=20, vmin=-10, vmax=10)
        plt.colorbar(s2, ax=ax2, label="Day-of Reaction (%)")
        z = np.polyfit(df["sentiment_score"], df["day_of_reaction_pct"], 1); xl = np.linspace(-1, 1, 100)
        ax2.plot(xl, np.poly1d(z)(xl), color=ACCENT_BLUE, lw=2, ls="--")
        corr = df["sentiment_score"].corr(df["day_of_reaction_pct"])
        ax2.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax2.transAxes, color=ACCENT_GOLD, fontsize=11, va="top")
        ax2.set_title("Sentiment vs Price Reaction", color="white", fontsize=13)
        ax2.set_xlabel("Sentiment Score", color="#8b949e"); ax2.set_ylabel("Day-of Reaction (%)", color="#8b949e")
        ax2.tick_params(colors="#8b949e")
        ax3 = axes[1][0]; ss2 = df.groupby("sector")["sentiment_score"].mean().sort_values()
        ax3.barh(ss2.index, ss2.values, color=[ACCENT_GREEN if v>0 else ACCENT_RED for v in ss2.values], edgecolor="none")
        ax3.axvline(0, color="white", lw=1, ls="--", alpha=0.5)
        ax3.set_title("Avg Sentiment by Sector", color="white", fontsize=13)
        ax3.set_xlabel("Mean Sentiment", color="#8b949e"); ax3.tick_params(colors="#8b949e")
        ax4 = axes[1][1]; hm = pd.crosstab(df["sentiment_label"], df["surprise_class"], normalize="index")
        hm = hm.reindex(index=lo,
                        columns=["Strong Miss","Moderate Miss","In-Line","Moderate Beat","Strong Beat"],
                        fill_value=0)
        sns.heatmap(hm, ax=ax4, cmap="RdYlGn", annot=True, fmt=".0%",
                    linewidths=0.5, linecolor="#30363d",
                    cbar_kws={"label":"Proportion"}, annot_kws={"size":9,"color":"white"})
        ax4.set_title("Sentiment to Surprise Class Heatmap", color="white", fontsize=13)
        ax4.tick_params(colors="#8b949e", rotation=30)
        ax4.set_xlabel("Surprise Class", color="#8b949e"); ax4.set_ylabel("Sentiment", color="#8b949e")
        fig.tight_layout(); self._save(fig, "chart_03_sentiment_analysis.png")

    def plot_pead(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Post-Earnings Announcement Drift (PEAD)", fontsize=18, fontweight="bold", color="white", y=1.02)
        order  = ["Strong Miss","Moderate Miss","In-Line","Moderate Beat","Strong Beat"]
        colors = [ACCENT_RED,"#f4a37d",ACCENT_GOLD,"#88d8a3",ACCENT_GREEN]
        periods = ["day_of_reaction_pct","drift_3d_pct","drift_10d_pct","drift_30d_pct"]
        plabels = ["Day 0","3-Day","10-Day","30-Day"]
        ax = axes[0]; x = np.arange(len(periods)); w = 0.15
        for i, (sc, c) in enumerate(zip(order, colors)):
            means = [df[df["surprise_class"]==sc][p].mean() for p in periods]
            ax.bar(x+(i-2)*w, means, w, label=sc, color=c, alpha=0.85, edgecolor="none")
        ax.axhline(0, color="white", lw=0.8, alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels(plabels, color="#8b949e", fontsize=9)
        ax.set_ylabel("Avg Return (%)", color="#8b949e")
        ax.set_title("Cumulative Drift by Surprise Class", color="white", fontsize=13)
        ax.legend(fontsize=8, framealpha=0.3); ax.tick_params(colors="#8b949e")
        ax2 = axes[1]; gg = ["Raised","Maintained","Lowered"]; gc = [ACCENT_GREEN,ACCENT_GOLD,ACCENT_RED]
        for pi, (period, _) in enumerate(zip(["abnormal_return_1d","abnormal_return_10d"], ["1D","10D"])):
            for i, (guid, gcol) in enumerate(zip(gg, gc)):
                vals = df[df["guidance"]==guid][period].values
                bp2 = ax2.boxplot(vals, positions=[pi*4+i], widths=0.7, patch_artist=True,
                                  medianprops=dict(color="white", lw=2))
                for patch in bp2["boxes"]: patch.set_facecolor(gcol); patch.set_alpha(0.75)
        ax2.axhline(0, color="white", lw=0.8, alpha=0.4)
        ax2.set_xticks([1, 5])
        ax2.set_xticklabels(["1-Day Abnormal Return","10-Day Abnormal Return"], color="#8b949e", fontsize=10)
        patches = [mpatches.Patch(color=c, label=g) for c, g in zip(gc, gg)]
        ax2.legend(handles=patches, fontsize=9, framealpha=0.3)
        ax2.set_title("Abnormal Returns by Guidance", color="white", fontsize=13)
        ax2.set_ylabel("Abnormal Return (%)", color="#8b949e"); ax2.tick_params(colors="#8b949e")
        fig.tight_layout(); self._save(fig, "chart_04_post_earnings_drift.png")

    def plot_sector(self, df, ss):
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle("Sector-Level Earnings Intelligence", fontsize=18, fontweight="bold", color="white", y=1.02)
        ax = axes[0]
        metrics = ["avg_eps_surprise","beat_rate","avg_reaction","avg_sentiment","avg_eqs","guidance_raise_rate"]
        mlabels = ["Avg EPS Surprise","Beat Rate","Avg Reaction","Avg Sentiment","Avg EQS","Guidance Raise Rate"]
        hm = ss.set_index("sector")[metrics]; hn = (hm - hm.mean()) / hm.std()
        sns.heatmap(hn.T, ax=ax, cmap="RdYlGn", annot=hm.T.round(2), fmt="g",
                    linewidths=0.5, linecolor="#30363d",
                    annot_kws={"size":9,"color":"white"}, cbar_kws={"label":"Z-Score"})
        ax.set_title("Sector Performance Z-Score", color="white", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", color="#8b949e")
        ax.set_yticklabels(mlabels, rotation=0, color="#8b949e"); ax.set_xlabel(""); ax.set_ylabel("")
        ax2 = axes[1]
        pal = {"Technology":ACCENT_BLUE,"Finance":ACCENT_GOLD,"Healthcare":ACCENT_TEAL,
               "Consumer":ACCENT_PURPLE,"Energy":"#e6774a","Industrials":"#8b949e"}
        ts2 = EarningsSurpriseAnalytics().ticker_stats(df); top20 = ts2.nlargest(20, "avg_eqs")
        for _, row in top20.iterrows():
            ax2.scatter(row["avg_eps_surprise"], row["avg_reaction"], s=row["n"]*8,
                        c=pal.get(row["sector"],"white"), alpha=0.8, edgecolors="white", lw=0.5)
            ax2.annotate(row["ticker"], (row["avg_eps_surprise"], row["avg_reaction"]),
                         fontsize=7, color="white", alpha=0.85, xytext=(3,3), textcoords="offset points")
        ax2.axhline(0, color="white", lw=0.8, alpha=0.4); ax2.axvline(0, color="white", lw=0.8, alpha=0.4)
        ax2.set_title("Top-20 Tickers: EPS Surprise vs Reaction", color="white", fontsize=12)
        ax2.set_xlabel("Avg EPS Surprise (%)", color="#8b949e")
        ax2.set_ylabel("Avg Day-of Reaction (%)", color="#8b949e"); ax2.tick_params(colors="#8b949e")
        patches = [mpatches.Patch(color=c, label=s) for s, c in pal.items()]
        ax2.legend(handles=patches, fontsize=8, framealpha=0.3, loc="upper left")
        fig.tight_layout(); self._save(fig, "chart_05_sector_heatmap.png")

    def plot_ml(self, ml_results, fi):
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle("ML Signal Prediction Performance", fontsize=18, fontweight="bold", color="white", y=1.02)
        mc = {"Logistic Regression":ACCENT_BLUE,"Random Forest":ACCENT_GREEN,"Gradient Boosting":ACCENT_GOLD}
        ax = axes[0]
        for name, res in ml_results.items():
            fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
            auc_lbl = res["test_auc"]
            ax.plot(fpr, tpr, color=mc.get(name,"white"), lw=2, label=f"{name} (AUC={auc_lbl:.3f})")
        ax.plot([0,1],[0,1], color="#8b949e", lw=1, ls="--", alpha=0.5)
        ax.set_title("ROC Curves", color="white", fontsize=13)
        ax.set_xlabel("False Positive Rate", color="#8b949e"); ax.set_ylabel("True Positive Rate", color="#8b949e")
        ax.legend(fontsize=9, framealpha=0.3); ax.tick_params(colors="#8b949e")
        ax2 = axes[1]; names = list(ml_results.keys()); x = np.arange(len(names))
        ax2.bar(x-0.2, [ml_results[n]["cv_auc"] for n in names], 0.35, label="CV AUC", color=ACCENT_BLUE, alpha=0.85)
        ax2.bar(x+0.2, [ml_results[n]["test_auc"] for n in names], 0.35, label="Test AUC", color=ACCENT_GREEN, alpha=0.85)
        ax2.set_ylim(0.48, 0.70); ax2.axhline(0.5, color="white", lw=1, ls="--", alpha=0.4)
        ax2.set_xticks(x); ax2.set_xticklabels([n.replace(" ","\n") for n in names], color="#8b949e", fontsize=9)
        ax2.set_ylabel("ROC-AUC", color="#8b949e"); ax2.set_title("AUC Model Tournament", color="white", fontsize=13)
        ax2.legend(fontsize=9); ax2.tick_params(colors="#8b949e")
        ax3 = axes[2]
        if not fi.empty:
            top = fi.head(8)
            rl = {"eps_z_score":"EPS Z-Score","rev_z_score":"Rev Z-Score","guidance_score":"Guidance",
                  "sentiment_score":"Sentiment","nlp_confidence":"NLP Conf","n_sentiment_words":"Sent Words",
                  "pre_earnings_drift":"Pre-Drift","sec_enc":"Sector"}
            labels = [rl.get(f, f) for f in top["feature"]]
            cfi = [ACCENT_GREEN if v>top["importance"].median() else ACCENT_BLUE for v in top["importance"]]
            ax3.barh(labels[::-1], top["importance"].values[::-1], color=cfi[::-1], edgecolor="none")
            ax3.set_title("Feature Importance (Random Forest)", color="white", fontsize=13)
            ax3.set_xlabel("Importance", color="#8b949e"); ax3.tick_params(colors="#8b949e")
        fig.tight_layout(); self._save(fig, "chart_06_ml_performance.png")

    def plot_signal_dash(self, df):
        fig, axes = plt.subplots(2, 2, figsize=(18, 13))
        fig.suptitle("Composite Earnings Signal Dashboard", fontsize=18, fontweight="bold", color="white")
        sc_c = {"Strong Buy":ACCENT_GREEN,"Buy":"#88d8a3","Hold":ACCENT_GOLD,
                "Sell":"#f4a37d","Strong Sell":ACCENT_RED}
        so = ["Strong Sell","Sell","Hold","Buy","Strong Buy"]
        ax = axes[0][0]; counts = df["signal"].value_counts().reindex(so, fill_value=0)
        bars = ax.bar(counts.index, counts.values, color=[sc_c[s] for s in counts.index], edgecolor="none")
        ax.set_title("Signal Distribution", color="white", fontsize=13)
        ax.set_xlabel("Signal", color="#8b949e"); ax.set_ylabel("Events", color="#8b949e")
        ax.tick_params(colors="#8b949e", axis="x", rotation=20)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    f"{int(bar.get_height()):,}", ha="center", fontsize=9, color="white")
        ax2 = axes[0][1]; ar = df.groupby("signal")["day_of_reaction_pct"].mean().reindex(so)
        ax2.bar(ar.index, ar.values, color=[sc_c[s] for s in ar.index], edgecolor="none")
        ax2.axhline(0, color="white", lw=0.8, alpha=0.4)
        ax2.set_title("Avg Reaction by Signal", color="white", fontsize=13)
        ax2.set_xlabel("Signal", color="#8b949e"); ax2.set_ylabel("Avg Reaction (%)", color="#8b949e")
        ax2.tick_params(colors="#8b949e", axis="x", rotation=20)
        ax3 = axes[1][0]
        for sig in so:
            vals = df[df["signal"]==sig]["earnings_quality_score"].dropna()
            if len(vals)>0: ax3.hist(vals, bins=25, alpha=0.65, color=sc_c[sig], label=sig, edgecolor="none")
        ax3.set_title("EQS Distribution by Signal", color="white", fontsize=13)
        ax3.set_xlabel("EQS", color="#8b949e"); ax3.set_ylabel("Frequency", color="#8b949e")
        ax3.legend(fontsize=9, framealpha=0.3); ax3.tick_params(colors="#8b949e")
        ax4 = axes[1][1]; hr = df.groupby("signal")["positive_outcome"].mean().reindex(so) * 100
        ax4.bar(hr.index, hr.values, color=[sc_c[s] for s in hr.index], edgecolor="none")
        ax4.axhline(50, color="white", lw=1, ls="--", alpha=0.6); ax4.set_ylim(0, 100)
        ax4.set_title("Signal Hit Rate (%)", color="white", fontsize=13)
        ax4.set_xlabel("Signal", color="#8b949e"); ax4.set_ylabel("Hit Rate (%)", color="#8b949e")
        ax4.tick_params(colors="#8b949e", axis="x", rotation=20)
        for i, (idx, val) in enumerate(hr.items()):
            ax4.text(i, val+1, f"{val:.1f}%", ha="center", fontsize=9, color="white")
        fig.tight_layout(); self._save(fig, "chart_07_composite_signal.png")


def generate_report(df, ss, ml_results, fi, output_path):
    ts      = datetime.now().strftime("%B %d, %Y at %H:%M")
    best_s  = ss.loc[ss["avg_eqs"].idxmax(), "sector"]
    worst_s = ss.loc[ss["avg_eqs"].idxmin(), "sector"]
    br      = df["surprise_class"].isin(["Strong Beat","Moderate Beat"]).mean() * 100
    avgs    = df["sentiment_score"].mean()
    best_m  = max(ml_results, key=lambda k: ml_results[k]["test_auc"])
    best_a  = ml_results[best_m]["test_auc"]
    sb      = df[df["signal"]=="Strong Buy"]; sss = df[df["signal"]=="Strong Sell"]
    buy_hr  = sb["positive_outcome"].mean()*100 if len(sb)>0 else 0
    sell_hr = (1-sss["positive_outcome"].mean())*100 if len(sss)>0 else 0
    fi_top  = fi.iloc[0]["feature"] if not fi.empty else "eps_z_score"
    fi_name = {"eps_z_score":"EPS Surprise Z-Score","rev_z_score":"Revenue Z-Score",
               "guidance_score":"Guidance Action","sentiment_score":"NLP Sentiment Score",
               "nlp_confidence":"NLP Confidence","pre_earnings_drift":"Pre-Earnings Drift"}.get(fi_top, fi_top)
    sig_ord   = ["Strong Buy","Buy","Hold","Sell","Strong Sell"]
    n_tickers = len(df["ticker"].unique())
    n_sectors = df["sector"].nunique()
    n_events  = len(df)
    sec_rows = []
    for _, row in ss.sort_values("avg_eqs", ascending=False).iterrows():
        sec = row["sector"]; br2 = row["beat_rate"]; esp = row["avg_eps_surprise"]
        rxn = row["avg_reaction"]; eqs = row["avg_eqs"]; grr = row["guidance_raise_rate"]
        sec_rows.append(f"| {sec} | {br2:.0%} | {esp:+.1f}% | {rxn:+.2f}% | {eqs:+.3f} | {grr:.0%} |")
    sec_tbl = "\n".join(sec_rows)
    sig_rows = []
    for sig in sig_ord:
        g = df[df["signal"]==sig]
        if len(g) == 0: continue
        ng = len(g); eqs2 = g["earnings_quality_score"].mean(); sent = g["sentiment_score"].mean()
        rxn2 = g["day_of_reaction_pct"].mean(); hr = g["positive_outcome"].mean()*100
        sig_rows.append(f"| {sig} | {ng:,} | {eqs2:+.3f} | {sent:+.3f} | {rxn2:+.2f}% | {hr:.1f}% |")
    sig_tbl = "\n".join(sig_rows)
    ml_rows = []
    for n2, res in ml_results.items():
        cv2 = res["cv_auc"]; ta = res["test_auc"]
        ml_rows.append(f"| {n2} | {cv2:.4f} | {ta:.4f} |")
    ml_tbl = "\n".join(ml_rows)
    sc_order2 = ["Strong Beat","Moderate Beat","In-Line","Moderate Miss","Strong Miss"]
    sc_rows2 = []
    for sc in sc_order2:
        g2 = df[df["surprise_class"]==sc]; nc = len(g2); pct2 = nc/n_events*100
        rxn3 = g2["day_of_reaction_pct"].mean()
        sc_rows2.append(f"| {sc} | {nc:,} | {pct2:.1f}% | {rxn3:+.2f}% |")
    sc_tbl = "\n".join(sc_rows2)
    text_lines = [
        "# STOCK EARNINGS SURPRISE AND SENTIMENT INTELLIGENCE REPORT",
        "",
        f"**Generated:** {ts}",
        "**Author:** Vivek Kumar | Data Analytics Portfolio",
        f"**Coverage:** {n_tickers} Tickers | {n_sectors} Sectors | {n_events:,} Events",
        "",
        "---",
        "## Executive Summary",
        "",
        "| KPI | Value |",
        "|-----|-------|",
        f"| Total Events | {n_events:,} |",
        f"| Overall Beat Rate | {br:.1f}% |",
        f"| Avg Sentiment | {avgs:+.3f} |",
        f"| Best Sector EQS | {best_s} |",
        f"| Weakest Sector EQS | {worst_s} |",
        f"| Best ML Model | {best_m} AUC={best_a:.4f} |",
        f"| Strong Buy Hit Rate | {buy_hr:.1f}% |",
        f"| Strong Sell Hit Rate | {sell_hr:.1f}% |",
        "",
        "---",
        "## Earnings Surprise Analysis",
        "| Surprise Class | Count | % | Avg Day-of Reaction |",
        "|---|---|---|---|",
        sc_tbl,
        "",
        "---",
        "## Sector Scorecard",
        "| Sector | Beat Rate | Avg EPS Surprise | Avg Reaction | Avg EQS | Guidance Raise |",
        "|---|---|---|---|---|---|",
        sec_tbl,
        "",
        "---",
        "## Composite Signal Performance",
        "| Signal | Events | Avg EQS | Avg Sentiment | Day-of Reaction | Hit Rate |",
        "|---|---|---|---|---|---|",
        sig_tbl,
        "",
        "---",
        "## ML Model Tournament",
        "| Model | CV AUC | Test AUC |",
        "|---|---|---|",
        ml_tbl,
        "",
        f"**Champion:** {best_m} AUC={best_a:.4f} | **Top Feature:** {fi_name}",
        "",
        "---",
        "## Architecture",
        "",
        "- EarningsDataEngine: Multi-stock realistic earnings simulation",
        "- FinancialSentimentEngine: Loughran-McDonald NLP with VADER-style weighting",
        "- EarningsSurpriseAnalytics: Z-score normalization, EQS, composite signal",
        "- MLEngine: Model tournament (LR / RF / GBM) with 5-fold CV",
        "- VizEngine: 7 dark-theme professional financial charts",
        "",
        "---",
        "## Portfolio CV Relevance",
        "",
        "| Skill Demonstrated | Industry Signal |",
        "|---|---|",
        "| Financial NLP / Sentiment Analysis | Quant Research, FinTech Analytics |",
        "| Earnings Surprise Quantification | Investment Banking, Equity Research |",
        "| ML Signal Generation | Quantitative Finance, Hedge Funds |",
        "| Sector-Level Portfolio Analytics | Asset Management |",
        "| PEAD Anomaly Quantification | Academic / Research Finance |",
        "",
        "---",
        "",
        "*Vivek Kumar - Stock Earnings Sentiment Intelligence | Data Analytics Portfolio*",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))
    print(f"  [Report] Saved: {os.path.basename(output_path)}")


def run_full_pipeline():
    print("\n" + "="*80)
    print("  STOCK EARNINGS SURPRISE AND SENTIMENT INTELLIGENCE PLATFORM")
    print("  Author: Vivek Kumar | Portfolio Project")
    print("="*80)
    t0 = time.time()
    print("\n[1/6] Generating earnings universe...")
    df_raw   = EarningsDataEngine.generate(n=2000, seed=42)
    n_events = len(df_raw); n_tick = df_raw["ticker"].nunique(); n_sec = df_raw["sector"].nunique()
    print(f"  {n_events:,} events | {n_tick} tickers | {n_sec} sectors")
    df_raw.to_csv(os.path.join(DATA_DIR, "earnings_universe.csv"), index=False)
    print("\n[2/6] Running NLP sentiment scoring...")
    df_scored = FinancialSentimentEngine().score_df(df_raw)
    print("\n[3/6] Computing earnings quality scores and signals...")
    ana      = EarningsSurpriseAnalytics()
    df_final = ana.compute(df_scored)
    ss       = ana.sector_stats(df_final)
    ticker_s = ana.ticker_stats(df_final)
    df_final.to_csv(os.path.join(DATA_DIR, "earnings_scored.csv"), index=False)
    ss.to_csv(os.path.join(DATA_DIR, "sector_stats.csv"), index=False)
    ticker_s.to_csv(os.path.join(DATA_DIR, "ticker_stats.csv"), index=False)
    print("\n[4/6] Training ML signal prediction models...")
    ml = MLEngine(); ml_results, feats = ml.train(df_final); fi = ml.feature_importance(feats)
    print("\n[5/6] Generating visualizations...")
    viz = VizEngine(ARTIFACTS_DIR)
    viz.plot_surprise_dist(df_final); viz.plot_reaction_scatter(df_final)
    viz.plot_sentiment(df_final); viz.plot_pead(df_final)
    viz.plot_sector(df_final, ss); viz.plot_ml(ml_results, fi)
    viz.plot_signal_dash(df_final)
    print("\n[6/6] Generating intelligence report...")
    report_path = os.path.join(BASE_DIR, "EARNINGS_INTELLIGENCE_REPORT.md")
    generate_report(df_final, ss, ml_results, fi, report_path)
    elapsed = time.time() - t0; n_charts = len(viz.saved); n_rows = len(df_final)
    print("\n" + "="*80)
    print(f"  Pipeline Complete in {elapsed:.1f}s | {n_charts} charts generated")
    print(f"  Report : EARNINGS_INTELLIGENCE_REPORT.md")
    print(f"  Data   : data/earnings_scored.csv ({n_rows:,} rows)")
    print("="*80 + "\n")
    return df_final, ss, ml_results, fi


if __name__ == "__main__":
    run_full_pipeline()

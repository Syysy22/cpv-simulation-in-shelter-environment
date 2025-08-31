import os, re, pickle
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

# --------------------------- file --------------------------- #
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data")) 
STORE_PATH = os.path.join(DATA_DIR, "cpv_vax.pkl")   # <- single combined store

# --------------------------- config -------------------------- #
STAFF_ROUNDS_TARGET = 4       # choose staff clear cadence to show
ENV_DAYS_FILTER = None        # e.g. 7; None = auto-pick common value
EXCLUDE_ZERO_COVERAGE = False  # <-- keep 0% coverage runs (baseline)

COLOR_BY_COV = {0:"black", 25:"tab:orange", 50:"tab:blue", 75:"tab:green", 100:"tab:purple"}

# capacity lines
KENNEL_CAPACITY = 74
TOTAL_STAFF = 14

# --- Regime definitions by (P_DIRECT, P_ENV).
REGIMES = {
    "High":         (0.001, 0.0005),
    "Intermediate": (0.0005, 0.0003),
    "Low":          (0.0003, 0.0001),
}
TOL = 1e-12  # float equality tolerance

# ----------------------- helpers ----------------------------- #
def load_store(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Did you run infection_sim_exp_4.py?")
    with open(path, "rb") as f:
        st = pickle.load(f)
    if not isinstance(st, dict) or not st:
        raise ValueError(f"{path} empty or not a dict")
    return st

def get_clean_days(d, lbl=None):
    if isinstance(d, dict) and "clean_days" in d: return int(d["clean_days"])
    if lbl:
        m = re.search(r"env(\d+)d", lbl)
        if m: return int(m.group(1))
    return None

def get_staff_rounds(d, lbl=None):
    if isinstance(d, dict) and "staff_clear_rounds" in d: return int(d["staff_clear_rounds"])
    if lbl:
        m = re.search(r"staff(\d+)r", lbl)
        if m: return int(m.group(1))
    return None

def get_vax_pct(d, lbl=None):
    if isinstance(d, dict) and "vax_coverage" in d:
        return int(round(100*float(d["vax_coverage"])))
    if lbl:
        m = re.search(r"_vax(\d+)", lbl)
        if m: return int(m.group(1))
    return None

def rates_tuple(d):
    r = d.get("rates", {})
    return (float(r.get("direct", np.nan)), float(r.get("env", np.nan)))

def regime_name_from_rates(rt):
    for name, want in REGIMES.items():
        if abs(rt[0]-want[0]) <= TOL and abs(rt[1]-want[1]) <= TOL:
            return name
    return None 

# ----------------------- load & organize ---------------------- #
store = load_store(STORE_PATH)

# group labels by regime
by_regime = defaultdict(list)
for lbl, chunk in store.items():
    if not isinstance(chunk, dict):
        continue
    rn = regime_name_from_rates(rates_tuple(chunk))
    if rn:
        by_regime[rn].append(lbl)

if not by_regime:
    raise ValueError("No entries matched the known High/Intermediate/Low rate pairs in cpv_vax.pkl.")

# choose env_days
def env_of(lbl):
    return get_clean_days(store[lbl], lbl)

if ENV_DAYS_FILTER is None:
    common_env = None
    for rn, lbls in by_regime.items():
        envs = {env_of(l) for l in lbls if env_of(l) is not None}
        common_env = envs if common_env is None else (common_env & envs)
    if not common_env:
        raise ValueError("No common environment interval across regimes in cpv_vax.pkl.")
    # pick most frequent (tie -> smallest)
    counts = Counter(env_of(l) for lbls in by_regime.values() for l in lbls if env_of(l) in common_env)
    env_days_chosen = sorted(common_env, key=lambda d:(-counts[d], d))[0]
else:
    env_days_chosen = int(ENV_DAYS_FILTER)

# filter labels to selected env & staff rounds; collect coverage levels
labels_by_regime = {}
cov_sets = []
for rn, lbls in by_regime.items():
    keep = [l for l in lbls
            if get_staff_rounds(store[l], l) == STAFF_ROUNDS_TARGET
            and env_of(l) == env_days_chosen
            and get_vax_pct(store[l], l) is not None]
    if not keep:
        raise ValueError(f"{rn}: no runs for env={env_days_chosen}d, staff={STAFF_ROUNDS_TARGET}r.")
    labels_by_regime[rn] = keep
    cov_sets.append({get_vax_pct(store[l], l) for l in keep})

common_cov = set.intersection(*cov_sets) if cov_sets else set()
if EXCLUDE_ZERO_COVERAGE:
    common_cov = {p for p in common_cov if p and p > 0}
if not common_cov:
    raise ValueError("No common (>0%) coverage levels across regimes in cpv_vax.pkl.")
cov_pcts_to_plot = sorted(common_cov)

# align horizons
any_lbl = next(iter(labels_by_regime[next(iter(labels_by_regime))]))
common_max_day = min(int(store[l]["max_day"]) for lbls in labels_by_regime.values() for l in lbls)
common_n_rounds = min(int(store[l]["n_rounds"]) for lbls in labels_by_regime.values() for l in lbls)
x_days   = np.arange(common_max_day+1)
x_rounds = np.arange(common_n_rounds)

# ----------------------- plotting ----------------------------- #
def plot_small_multiples(metric_key, ylabel, title, x_is_rounds=False, hline=None, hline_label=None, ylim_min=0):
    x = x_rounds if x_is_rounds else x_days
    L = common_n_rounds if x_is_rounds else (common_max_day+1)

    regimes_order = ["High","Intermediate","Low"]
    regimes_order = [r for r in regimes_order if r in labels_by_regime]
    fig, axes = plt.subplots(1, len(regimes_order), figsize=(13, 4.2), sharex=True, sharey=True)

    global_max = 0.0
    for ax, rn in zip(axes, regimes_order):
        lbls = [l for l in labels_by_regime[rn] if get_vax_pct(store[l], l) in cov_pcts_to_plot]
        lbls.sort(key=lambda k: get_vax_pct(store[k], k))
        for l in lbls:
            pct = get_vax_pct(store[l], l)
            y = np.asarray(store[l][metric_key])[:L]
            label = "Baseline (0% Coverage)" if pct == 0 else f"{pct}% Coverage"
            ax.plot(x, y, label=label, color=COLOR_BY_COV.get(pct))
            if np.isfinite(y).any():
                global_max = max(global_max, float(np.nanmax(y)))

        if hline is not None:
            ax.axhline(hline, ls="--", color="grey", lw=1, label=(hline_label if rn == regimes_order[0] else None))

        ax.set_title(f"{rn} Transmission")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Round" if x_is_rounds else "Day")

    axes[0].set_ylabel(ylabel)
    ylim_top = max(np.ceil(global_max*1.05/5)*5, ylim_min+1)
    for ax in axes:
        ax.set_ylim(ylim_min, ylim_top)

    # one legend
    handles, labels = [], []
    for p in cov_pcts_to_plot:
        lbl = "Baseline (0% Coverage)" if p == 0 else f"{p}% Coverage"
        h, = axes[0].plot([], [], color=COLOR_BY_COV.get(p), label=lbl)
        handles.append(h)
        labels.append(lbl)
    if hline is not None:
        ref, = axes[0].plot([], [], ls="--", color="grey", label=hline_label or "")
        handles.append(ref)
        labels.append(hline_label or "")
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False)

    fig.suptitle(title + f" (env={env_days_chosen}d, staff={STAFF_ROUNDS_TARGET}r)", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0.10, 1, 0.92])
    plt.show()

# --------------------------- plots --------------------------- #
plot_small_multiples(
    metric_key="avg_total_infected_per_day",
    ylabel="Number Of Infected Dogs",
    title="Number Of Infected Dogs Per Day",
    x_is_rounds=False,
    hline=KENNEL_CAPACITY, hline_label="Kennel Capacity", ylim_min=0
)

plot_small_multiples(
    metric_key="avg_contaminated_per_day",
    ylabel="Number Of Contaminated Nodes",
    title="Number Of Contaminated Nodes Per Day",
    x_is_rounds=False, ylim_min=0
)

plot_small_multiples(
    metric_key="avg_staff_per_round",
    ylabel="Number Of Staff Carrying CPV",
    title="Staff Infection Per Round",
    x_is_rounds=True,
    hline=TOTAL_STAFF, hline_label="Total Staff", ylim_min=0
)

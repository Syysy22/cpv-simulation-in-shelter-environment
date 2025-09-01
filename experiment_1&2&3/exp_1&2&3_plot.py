import os, re, pickle
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

# ---------- Paths ----------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
STORE_PATH = os.path.join(DATA_DIR, "cpv_env_staff_combo.pkl")

# ---------- Config ----------
KENNEL_CAPACITY = 74
TOTAL_STAFF     = 14
USE_COMPACT_LABELS = True

# pre-filter which combos to show, set these; None = show all found
ENV_DAYS_FILTER = None        # {1,3,7} or None
STAFF_R_FILTER  = None        # {1,2,4} or None

# Regime definitions by (P_DIRECT, P_ENV)
REGIMES = {
    "High":         (0.001, 0.0005),
    "Intermediate": (0.0005, 0.0003),
    "Low":          (0.0003, 0.0001),
}
TOL = 1e-12  # float equality tolerance for rate matching

COLOR_BY_COMBO = {
    (1,1): "tab:blue",   (1,2): "tab:orange", (1,4): "tab:green",
    (3,1): "tab:red",    (3,2): "tab:brown",  (3,4): "tab:purple",
    (7,1): "tab:gray",   (7,2): "olive",      (7,4): "tab:cyan",
}

# ---------- Helpers ----------
def load_store(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Did you run infection_sim_exp_1&2&3.py?")
    with open(path, "rb") as f:
        st = pickle.load(f)
    if not isinstance(st, dict) or not st:
        raise ValueError(f"{path} empty or not a dict")
    return st

def rates_tuple(d):
    r = d.get("rates", {})
    return (float(r.get("direct", np.nan)), float(r.get("env", np.nan)))

def regime_name_from_rates(rt):
    for name, want in REGIMES.items():
        if abs(rt[0]-want[0]) <= TOL and abs(rt[1]-want[1]) <= TOL:
            return name
    return None

def get_env_days(entry, label):
    if isinstance(entry, dict) and "clean_days" in entry: return int(entry["clean_days"])
    m = re.search(r"env(\d+)d", label or "")
    return int(m.group(1)) if m else None

def get_staff_rounds(entry, label):
    if isinstance(entry, dict) and "staff_clear_rounds" in entry: return int(entry["staff_clear_rounds"])
    m = re.search(r"staff(\d+)r", label or "")
    return int(m.group(1)) if m else None

def combo_label(env_days, staff_rounds):
    if USE_COMPACT_LABELS:
        return f"E{env_days}d + S{staff_rounds}r"
    s_map = {1: "Every Round", 2: "Every 2 Rounds", 4: "Once Per Day (4 Rounds)"}
    e_txt = f"Environment: every {env_days} day{'s' if env_days != 1 else ''}"
    s_txt = f"Staff: {s_map.get(staff_rounds, f'every {staff_rounds} rounds')}"
    return f"{e_txt} + {s_txt}"

# ---------- Load & organise ----------
store = load_store(STORE_PATH)

# Group entries by regime -> list of labels
labels_by_regime = defaultdict(list)
for lbl, chunk in store.items():
    if not isinstance(chunk, dict) or "rates" not in chunk:
        continue 
    rn = regime_name_from_rates(rates_tuple(chunk))
    if rn:
        labels_by_regime[rn].append(lbl)

if not labels_by_regime:
    raise ValueError("No entries matched High/Intermediate/Low rate pairs in cpv_env_staff_combo.pkl.")

# restrict which combos to show
def env_of(l): return get_env_days(store[l], l)
def staff_of(l): return get_staff_rounds(store[l], l)

if ENV_DAYS_FILTER is not None:
    ENV_DAYS_FILTER = set(map(int, ENV_DAYS_FILTER))
if STAFF_R_FILTER is not None:
    STAFF_R_FILTER = set(map(int, STAFF_R_FILTER))

# Determine the set of combos present in each regime
combos_by_regime = {}
for rn, lbls in labels_by_regime.items():
    combos = []
    for l in lbls:
        e, s = env_of(l), staff_of(l)
        if e is None or s is None: continue
        if (ENV_DAYS_FILTER and e not in ENV_DAYS_FILTER) or (STAFF_R_FILTER and s not in STAFF_R_FILTER):
            continue
        combos.append((e, s))
    combos_by_regime[rn] = sorted(set(combos))

# Align horizons across selected labels (days/rounds lengths can differ)
selected_labels = []
for rn, lbls in labels_by_regime.items():
    for l in lbls:
        e, s = env_of(l), staff_of(l)
        if e is None or s is None: continue
        if (ENV_DAYS_FILTER and e not in ENV_DAYS_FILTER) or (STAFF_R_FILTER and s not in STAFF_R_FILTER):
            continue
        selected_labels.append(l)

if not selected_labels:
    raise ValueError("No labels remain after applying filters.")

common_max_day  = min(int(store[l]["max_day"])  for l in selected_labels)
common_n_rounds = min(int(store[l]["n_rounds"]) for l in selected_labels)
x_days   = np.arange(common_max_day+1)
x_rounds = np.arange(common_n_rounds)

# ---------- Plotting ----------
def plot_metric(metric_key, title, ylabel, x_is_rounds=False, refline=None, alpha=1.0):
    x = x_rounds if x_is_rounds else x_days
    L = common_n_rounds if x_is_rounds else (common_max_day+1)

    regimes_order = ["High", "Intermediate", "Low"]
    regimes_order = [r for r in regimes_order if r in labels_by_regime]
    fig, axes = plt.subplots(1, len(regimes_order), figsize=(13, 4.8), sharex=True, sharey=True)

    global_max = 0.0
    for ax, rn in zip(axes, regimes_order):
        # plot each combo present for this regime
        for combo in combos_by_regime.get(rn, []):
            # find the label matching this combo (env_days, staff_rounds)
            candidates = [l for l in labels_by_regime[rn]
                          if env_of(l) == combo[0] and staff_of(l) == combo[1]]
            if not candidates:
                continue
            lbl = sorted(candidates)[0]  # if multiple, pick one deterministically
            y = np.asarray(store[lbl].get(metric_key, []))[:L]
            if y.size == 0:
                continue
            ax.plot(
                x, y,
                color=COLOR_BY_COMBO.get(combo, "black"),
                lw=1.2,
                alpha=alpha,
                label=combo_label(*combo),
            )
            if np.isfinite(y).any():
                global_max = max(global_max, float(np.nanmax(y)))

        if refline is not None:
            ax.axhline(refline[0], ls="--", color="grey", lw=1, label=(refline[1] if rn == regimes_order[0] else None))

        ax.set_title(f"{rn} Transmission")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Round" if x_is_rounds else "Day")

    axes[0].set_ylabel(ylabel)
    ylim_top = max(np.ceil(global_max*1.05/5)*5, 1)
    for ax in axes:
        ax.set_ylim(0, ylim_top)

    # Build legend of only combos we actually plotted
    present = set()
    for rn in regimes_order:
        for combo in combos_by_regime.get(rn, []):
            present.add(combo)
    handles, labels = [], []
    for combo in sorted(present):
        h, = axes[0].plot([], [], color=COLOR_BY_COMBO.get(combo, "black"), lw=1.6, label=combo_label(*combo))
        handles.append(h); labels.append(combo_label(*combo))
    if refline is not None:
        ref, = axes[0].plot([], [], ls="--", color="grey", label=refline[1])
        handles.append(ref); labels.append(refline[1])

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        fontsize=8,
        title=("E = Environment Cleaning Interval (Days); "
               "S = Staff Decontamination Interval (Rounds)")
               if USE_COMPACT_LABELS else
               "Strategy (Environment cleaning + Staff decontamination)",
        title_fontsize=9,
        frameon=False,
        handlelength=2.0,
        columnspacing=0.9,
        labelspacing=0.35,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.show()

# 1) Infected dogs per day
plot_metric(
    metric_key="avg_total_infected_per_day",
    title="Number Of Infected Dogs Per Day",
    ylabel="Number Of Infected Dogs",
    x_is_rounds=False,
    refline=(KENNEL_CAPACITY, "Kennel Capacity"),
)

# 2) Contaminated nodes per day
plot_metric(
    metric_key="avg_contaminated_per_day",
    title="Number Of Contaminated Nodes Per Day",
    ylabel="Number Of Contaminated Nodes",
    x_is_rounds=False,
)

# 3) Staff infection per round
plot_metric(
    metric_key="avg_staff_per_round",
    title="Staff Infection Per Round",
    ylabel="Number Of Staff Carrying CPV",
    x_is_rounds=True,
    refline=(TOTAL_STAFF, "Total Staff"),
    alpha=0.55,
)

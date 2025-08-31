import os, time, math, random, pickle
import numpy as np
from collections import defaultdict

# ---------- Paths ----------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FULL_STATE_PATH   = os.path.join(DATA_DIR, "full_simulation.pkl")  # baseline snapshot
MASTER_SAVE_PATH  = os.path.join(DATA_DIR, "cpv_env_staff_combo.pkl")  # single output

# ========= Regime-specific settings =========
# High:         P_DIRECT=0.001,  P_ENV=0.0005, MASTER_SAVE_PATH="cpv_env_staff_combo_high.pkl"
# Intermediate: P_DIRECT=0.0005, P_ENV=0.0003, MASTER_SAVE_PATH="cpv_env_staff_combo_medium.pkl"
# Low:          P_DIRECT=0.0003, P_ENV=0.0001, MASTER_SAVE_PATH="cpv_env_staff_combo_low.pkl"
P_DIRECT = 0.0003        # set per regime
P_ENV    = 0.0001        # set per regime

# run these pairs (env_days, staff_clear_rounds):
COMBOS_TO_RUN = [(7,1), (7,2), (7,4)]

# ========= Sim constants =========
N_REPS = 50
ROUNDS_PER_DAY = 4
BASE_SEED = 12345
INCUBATION_MIN, INCUBATION_MAX = 1, 5  # days
DETECT_START   = 14
RECOVER_START  = 20

# ========= Load baseline =========
with open(FULL_STATE_PATH, "rb") as f:
    saved = pickle.load(f)

occupancy                   = saved["occupancy"]
round_start_times           = sorted(saved["round_start_times"])
original_dog_health         = saved["dog_health"]
original_staff_health       = saved["staff_health"]
original_node_contamination = saved["node_contamination"]

all_times = sorted(occupancy.keys())
max_day = all_times[-1] // 86400

# ========= Helper =========
def run_combo(env_days: int, staff_clear_rounds: int):
    clean_rounds = env_days * ROUNDS_PER_DAY

    daily_total_infected_matrix = np.zeros((N_REPS, max_day + 1))
    daily_infectious_matrix     = np.zeros((N_REPS, max_day + 1))
    daily_contaminated_matrix   = np.zeros((N_REPS, max_day + 1))
    total_dogs_simulated        = np.zeros(N_REPS)
    per_round_infected_staff_runs = []

    t0 = time.time()
    for sim in range(N_REPS):
        random.seed(BASE_SEED + sim)
        np.random.seed(BASE_SEED + sim)

        dog_health         = original_dog_health.copy()
        staff_health       = original_staff_health.copy()
        node_contamination = original_node_contamination.copy()

        exposure_time = {}
        detected_and_isolated = set()
        infectious_dogs = set()

        daily_total_infected_dogs = defaultdict(set)
        daily_currently_infectious_dogs = defaultdict(set)
        daily_contaminated_nodes = defaultdict(set)
        dog_replaced = defaultdict(int)

        per_round_staff_infected = np.zeros(len(round_start_times), dtype=int)
        current_round_infected_staff = set()

        # mark initially exposed
        for d in dog_health:
            if dog_health[d] == 1:
                exposure_time[d] = 0

        for t in all_times:
            # boundaries
            if t in round_start_times:
                r_idx = round_start_times.index(t)
                if r_idx > 0:
                    per_round_staff_infected[r_idx - 1] = len(current_round_infected_staff)
                    current_round_infected_staff.clear()
                if (r_idx % staff_clear_rounds) == 0:
                    staff_health = {i: 0 for i in staff_health}
                if r_idx > 0 and (r_idx % clean_rounds) == 0:
                    node_contamination = {i: 0 for i in node_contamination}

            day_index = t // 86400

            # incubation
            for d in sorted(exposure_time.keys()):
                if d not in infectious_dogs:
                    inc = (t - exposure_time[d]) / 86400
                    if inc >= random.randint(INCUBATION_MIN, INCUBATION_MAX):
                        infectious_dogs.add(d)

            for node in sorted(occupancy[t].keys()):
                occupants = occupancy[t][node]      

                # Consider only active dogs (not isolated)
                dogs_here  = sorted({id for kind, id in occupants
                                    if kind == "dog" and id not in detected_and_isolated})
                staff_here = sorted({id for kind, id in occupants if kind == "staff"})

                # --- Direct contact: dog↔staff and dog↔dog ---
                for d in dogs_here:
                    if d in infectious_dogs:
                        for s in staff_here:
                            if staff_health[s] == 0 and random.random() < P_DIRECT:
                                staff_health[s] = 1

                        for d2 in dogs_here:
                            if d != d2 and dog_health[d2] == 0 and random.random() < P_DIRECT:
                                    dog_health[d2] = 1
                                    exposure_time[d2] = t
                    
                    elif dog_health[d] == 0:
                        for s in staff_here:
                            if staff_health[s] == 1 and random.random() < P_DIRECT:
                                dog_health[d] = 1
                                exposure_time[d] = t
            

                # --- Environment transmission to unexposed ---
                for d in dogs_here:
                    if dog_health[d] == 0 and node_contamination.get(node, 0) == 1 and random.random() < P_ENV:
                        dog_health[d] = 1
                        exposure_time[d] = t

                for s in staff_here:
                    if staff_health[s] == 0 and node_contamination.get(node, 0) == 1 and random.random() < P_ENV:
                        staff_health[s] = 1

                # --- Environment contamination from infected ---
                if any(d in infectious_dogs for d in dogs_here) or any(staff_health.get(s, 0) == 1 for s in staff_here):
                    if random.random() < P_ENV:
                        node_contamination[node] = 1

                # Record node contamination for the day
                if node_contamination.get(node, 0) == 1:
                    daily_contaminated_nodes[day_index].add(node)
        
            # detection & recovery
            for d in sorted(infectious_dogs):
                days = (t - exposure_time[d]) / 86400
                if d not in detected_and_isolated and days >= DETECT_START:
                    p_detect = 1 / (1 + math.exp(-0.6 * (days - 17)))
                    if random.random() < p_detect:
                        detected_and_isolated.add(d)
                if days >= RECOVER_START:
                    p_rec = 1 / (1 + math.exp(-0.3 * (days - 27)))
                    if random.random() < p_rec:
                        dog_health[d] = 0
                        exposure_time.pop(d, None)
                        infectious_dogs.discard(d)
                        detected_and_isolated.discard(d)
                        dog_replaced[d] += 1

               
            for d in dog_health:
                if dog_health[d] == 1:
                    daily_total_infected_dogs[day_index].add(d)
            for d in infectious_dogs:
                if d not in detected_and_isolated:
                    daily_currently_infectious_dogs[day_index].add(d)

            current_round_infected_staff.update([i for i, v in staff_health.items() if v == 1])

        per_round_staff_infected[len(round_start_times) - 1] = len(current_round_infected_staff)

        # aggregate
        for day in range(max_day + 1):
            daily_total_infected_matrix[sim, day] = len(daily_total_infected_dogs.get(day, []))
            daily_infectious_matrix[sim, day]     = len(daily_currently_infectious_dogs.get(day, []))
            daily_contaminated_matrix[sim, day]   = len(daily_contaminated_nodes.get(day, []))

        unreplaced = set(dog_health) - set(dog_replaced)
        total_dogs_simulated[sim] = sum(dog_replaced[d] + 1 for d in dog_replaced) + len(unreplaced)
        per_round_infected_staff_runs.append(per_round_staff_infected)

    # means
    avg_contaminated = np.round(daily_contaminated_matrix.mean(axis=0)).astype(int)
    avg_total_inf    = np.round(daily_total_infected_matrix.mean(axis=0)).astype(int)
    avg_infectious   = np.round(daily_infectious_matrix.mean(axis=0)).astype(int)
    avg_total_dogs   = int(round(total_dogs_simulated.mean()))
    staff_mat        = np.vstack(per_round_infected_staff_runs)
    avg_staff_round  = staff_mat.mean(axis=0)

    label = f"env{env_days}d_staff{staff_clear_rounds}r"
    return label, {
        "label": label,
        "clean_days": env_days,
        "clean_rounds": env_days * ROUNDS_PER_DAY,
        "staff_clear_rounds": staff_clear_rounds,
        "max_day": int(max_day),
        "n_rounds": len(round_start_times),
        "avg_total_infected_per_day": avg_total_inf,
        "avg_infectious_per_day":     avg_infectious,
        "avg_contaminated_per_day":   avg_contaminated,
        "avg_staff_per_round":        avg_staff_round,
        "avg_total_dogs":             avg_total_dogs,
        "rates": {"direct": P_DIRECT, "env": P_ENV},
        "rounds_per_day": ROUNDS_PER_DAY,
        "n_reps": N_REPS,
        "elapsed_sec": time.time() - t0,
    }

# ---------- Run all combos, save once ----------
MASTER = {}  # fresh master for this run

for env_days, staff_rounds in COMBOS_TO_RUN:
    key = f"env{env_days}d_staff{staff_rounds}r"
    print(f"Running {key} …")
    lbl, res = run_combo(env_days, staff_rounds)
    MASTER[lbl] = res

# write a single file at the very end
with open(MASTER_SAVE_PATH, "wb") as f:
    pickle.dump(MASTER, f)
print(f"Done. Wrote {len(MASTER)} entries to {MASTER_SAVE_PATH}")

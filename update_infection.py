import pickle, random, math, numpy as np, matplotlib.pyplot as plt, time, matplotlib.animation as animation, networkx as nx
from collections import defaultdict
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Build the path to the simulation results inside the repo
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "data", "full_simulation.pkl")

# Load saved simulation data (with a clear error if missing)
if os.path.exists(data_path):
    with open(data_path, "rb") as f:
        saved = pickle.load(f)
    print(f"Loaded simulation data from {data_path}")
else:
    raise FileNotFoundError(
        f"full_simulation.pkl not found in {data_path}. "
        "Please generate it first by running movement_simulation.py."
    )

G = saved["graph"]
coord_to_index = saved["coord_to_index"]
index_to_coord = {v: k for k, v in coord_to_index.items()}

occupancy = saved["occupancy"]
round_start_times = sorted(saved["round_start_times"])
original_dog_health = saved["dog_health"]
original_staff_health = saved["staff_health"]
original_node_contamination = saved["node_contamination"]


# get all simulation timestamps and days
all_times = sorted(occupancy.keys())
max_day = all_times[-1] // 86400

# initialize matrices to hold daily infection and contamination data
daily_total_infected_matrix = np.zeros((50, max_day + 1))
daily_infectious_matrix = np.zeros((50, max_day + 1))
daily_contaminated_matrix = np.zeros((50, max_day + 1))
total_dogs_simulated = np.zeros(50)
contamination_frequency_per_day = defaultdict(lambda: defaultdict(int))  # day → node → count
per_round_infected_staff_runs = [] # store per-round staff metric across simulations

ROUNDS_PER_DAY = 4
CLEAN_DAYS = 7         # <- set to 1, 2, 3, or 7 for each run
CLEAN_ROUNDS = CLEAN_DAYS * ROUNDS_PER_DAY

STAFF_CLEAR_ROUNDS = 4

STRATEGY_LABEL = f"env{CLEAN_DAYS}d_staff{STAFF_CLEAR_ROUNDS}r"

BASE_SEED = 12345

start_time = time.time()
# Run simulation 50 times
for sim in range(50):
    random.seed(BASE_SEED + sim)
    np.random.seed(BASE_SEED + sim)

    dog_health = original_dog_health.copy()
    staff_health = original_staff_health.copy()
    node_contamination = original_node_contamination.copy()

    # --- Infection tracking ---
    exposure_time = {}  # dog_id → time of exposure
    detected_and_isolated = set()  # discovered and isolated dogs
    infectious_dogs = set()  # dog currently infectious
    recovered_dogs = set()
    daily_total_infected_dogs = defaultdict(set)  # day → set of exposed + infectious + discovered dogs
    daily_contaminated_nodes = defaultdict(set)  # day → set of contaminated node indices
    dog_replaced = defaultdict(int)   # count how many times each dog ID was replaced by a new intake
    daily_currently_infectious_dogs = defaultdict(set)  # Only infectious
    per_round_staff_infected = np.zeros(len(round_start_times), dtype=int) # Per-round: staff infected at any time within the round
    current_round_infected_staff = set()

    # Set exposure time = 0 for all initially exposed dogs
    for d in dog_health:
        if dog_health[d] == 1:
            exposure_time[d] = 0    

    # --- Infection simulation ---
    for t in sorted(occupancy.keys()):

        # Reset staff infection at start of each round
        if t in round_start_times:
            round_idx = round_start_times.index(t)
                        
            if round_idx > 0:       # Close out the previous round (skip for the very first round)
                per_round_staff_infected[round_idx - 1] = len(current_round_infected_staff)
                current_round_infected_staff.clear()
            
            if (round_idx % STAFF_CLEAR_ROUNDS) == 0:
                staff_health = {i: 0 for i in staff_health}
       
            if round_idx > 0 and (round_idx % CLEAN_ROUNDS == 0):    # Clear environment contamination every 28 rounds (once per week)
                node_contamination = {i: 0 for i in node_contamination}

        day_index = t // 86400  # convert seconds to day

        # Determine who has become infectious based on incubation
        for d in sorted(exposure_time.keys()):
            if d not in infectious_dogs:
                incubation_days = (t - exposure_time[d]) / 86400
                if incubation_days >= random.randint(1, 5):  # Incubation = 1–5 days
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
                        if staff_health[s] == 0 and random.random() < 0.0003:
                            staff_health[s] = 1

                    for d2 in dogs_here:
                        if d != d2 and dog_health[d2] == 0 and random.random() < 0.0003:
                                dog_health[d2] = 1
                                exposure_time[d2] = t
                
                elif dog_health[d] == 0:
                    for s in staff_here:
                        if staff_health[s] == 1 and random.random() < 0.0003:
                            dog_health[d] = 1
                            exposure_time[d] = t
          

            # --- Environment transmission to unexposed ---
            for d in dogs_here:
                if dog_health[d] == 0 and node_contamination.get(node, 0) == 1 and random.random() < 0.0001:
                    dog_health[d] = 1
                    exposure_time[d] = t

            for s in staff_here:
                if staff_health[s] == 0 and node_contamination.get(node, 0) == 1 and random.random() < 0.0001:
                    staff_health[s] = 1

            # --- Environment contamination from infected ---
            if any(d in infectious_dogs for d in dogs_here) or any(staff_health.get(s, 0) == 1 for s in staff_here):
                if random.random() < 0.0001:
                    node_contamination[node] = 1

            # Record node contamination for the day
            if node_contamination.get(node, 0) == 1:
                daily_contaminated_nodes[day_index].add(node)


         # --- Detection & isolation & recovery ---
        for d in sorted(infectious_dogs):
            days_exposed = (t - exposure_time[d]) / 86400

        # Detection after exposure (day 14+)
            if d not in detected_and_isolated and days_exposed >= 14:
                    p_detect = 1 / (1 + math.exp(-0.6 * (days_exposed - 17)))

                    if random.random() < p_detect:
                        detected_and_isolated.add(d)

        # Recovery after max 20 days of exposure
            if d not in recovered_dogs and days_exposed >= 20:
                    p_recover = 1 / (1 + math.exp(-0.3 * (days_exposed - 27)))

                    if random.random() < p_recover:
                        dog_health[d] = 0
                        recovered_dogs.add(d)

                        # Treat as a new intake
                        del exposure_time[d]  # remove previous exposure record
                        infectious_dogs.discard(d)
                        detected_and_isolated.discard(d)
                        recovered_dogs.discard(d)  # remove it again immediately
                        dog_replaced[d] += 1  # new intake count for this kennel


    # --- Daily infection tracking ---
        for d in dog_health:
            if dog_health[d] == 1:
                daily_total_infected_dogs[day_index].add(d)
        for d in infectious_dogs:
            if d not in detected_and_isolated:
                daily_currently_infectious_dogs[day_index].add(d)
        current_round_infected_staff.update([i for i, v in staff_health.items() if v == 1])

    # Close out the final round after finishing all timestamps
    per_round_staff_infected[len(round_start_times) - 1] = len(current_round_infected_staff)

    for day in range(max_day + 1):
        daily_total_infected_matrix[sim, day] = len(daily_total_infected_dogs.get(day, []))
        daily_infectious_matrix[sim, day] = len(daily_currently_infectious_dogs.get(day, []))
        daily_contaminated_matrix[sim, day] = len(daily_contaminated_nodes.get(day, []))  

    unreplaced_dogs = set(dog_health) - set(dog_replaced)
    total_dogs_simulated[sim] = sum(dog_replaced[d] + 1 for d in dog_replaced) + len(unreplaced_dogs)
    
    for day, nodes in daily_contaminated_nodes.items():
        for node in nodes:
            contamination_frequency_per_day[day][node] += 1

    # Save per-round staff metric for this run
    per_round_infected_staff_runs.append(per_round_staff_infected)

    print(f"Simulation {sim+1} finished.")


end_time = time.time()
elapsed = end_time - start_time
print(f"All 50 simulations completed in {elapsed:.2f} seconds.")


# Compute averages
average_daily_contaminated = np.round(daily_contaminated_matrix.mean(axis=0)).astype(int)
average_total_infected = np.round(daily_total_infected_matrix.mean(axis=0)).astype(int)
average_infectious = np.round(daily_infectious_matrix.mean(axis=0)).astype(int)
average_total_dogs = int(round(total_dogs_simulated.mean()))
infected_staff_runs_matrix = np.vstack(per_round_infected_staff_runs)
average_infected_staff_per_round = infected_staff_runs_matrix.mean(axis=0) 

x_days   = np.arange(max_day + 1)
x_rounds = np.arange(len(round_start_times))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_days, average_total_infected, label="Number Of Infected Dogs")
ax.axhline(y=74, linestyle="--", label="Total Dog Capacity")
ax.set_ylim(0, 80)
ax.set_title("Number Of Infected Dogs Per Day")
ax.set_xlabel("Day"); ax.set_ylabel("Number Of Infected Dogs")
ax.grid(True)
ax.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.9)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_days, average_infectious, label="Number Of Infectious Dogs")
ax.axhline(y=74, linestyle="--", label="Total Dog Capacity")
ax.set_ylim(0, 80)
ax.set_title("Number Of Infectious Dogs Per Day")
ax.set_xlabel("Day"); ax.set_ylabel("Number Of Infectious Dogs")
ax.grid(True)
ax.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.9)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_days, average_daily_contaminated)
ax.set_ylim(0, 150)
ax.set_title("Number Of Contaminated Nodes Per Day")
ax.set_xlabel("Day"); ax.set_ylabel("Number Of Contaminated Nodes")
ax.grid(True)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_rounds, average_infected_staff_per_round, label="Number Of Staff Carrying CPV")
ax.set_ylim(0, 20)
ax.axhline(y=14, linestyle='--', label='Total Number Of Staff')
ax.set_title("Staff Infection Per Round")
ax.set_xlabel("Round"); ax.set_ylabel("Number Of Staff Carrying CPV")
ax.grid(True)
ax.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.9)
plt.tight_layout(); plt.show()

# -------------heatmap---------------------
contamination_probability = {
    day: {node: count / 50 for node, count in node_counts.items()}
    for day, node_counts in contamination_frequency_per_day.items()
}

fig, ax = plt.subplots(figsize=(10, 8))
pos = {i: index_to_coord[i] for i in G.nodes}

def update(day):
    ax.clear()
    ax.set_title(f"Day {day} - Contaminated Node Heatmap")
    ax.axis('off')

    node_color = []
    for node in G.nodes():
        freq = contamination_frequency_per_day.get(day, {}).get(node, 0)
        intensity = freq / 50
        node_color.append(intensity)

    nx.draw_networkx_nodes(G, pos, node_size=30, node_color=node_color, cmap='hot', vmin=0, vmax=1, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5, alpha=0.5, ax=ax)


# Create colorbar based on same colormap and scale
sm = ScalarMappable(cmap='hot', norm=Normalize(vmin=0, vmax=1))
sm.set_array([])

# Add colorbar to the plot
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
cbar.set_label('Contamination Probability (0-1)', fontsize=10)

ani = animation.FuncAnimation(fig, update, frames=range(max_day + 1), interval=500)
ani.save("contaminated_nodes_heatmap.gif", writer="pillow", dpi=200)


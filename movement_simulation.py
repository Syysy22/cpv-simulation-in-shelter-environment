import os, sys, time, pickle, random, matplotlib.pyplot as plt, networkx as nx, geopandas as gpd
from collections import defaultdict
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "data", "paths.shp")
gdf = gpd.read_file(data_path)

nodes = set()
for line in gdf.geometry:
    coords = list(line.coords)
    for pt in coords:
        nodes.add(pt)
nodes = list(nodes)
coord_to_index = {coord: idx for idx, coord in enumerate(nodes)}

G = nx.Graph()
for line in gdf.geometry:
    coords = list(line.coords)
    for i in range(len(coords) - 1):
        u = coord_to_index[coords[i]]
        v = coord_to_index[coords[i + 1]]
        G.add_edge(u, v, weight=1)  

def generate_random_walk(subgraph, start_node, total_steps=180):
    path = [start_node]
    current = start_node
    prev = None
    for i in range(total_steps):
        neighbors = list(subgraph.neighbors(current))
        if prev is not None and prev in neighbors and len(neighbors) > 1:
            neighbors.remove(prev)  # avoid immediate reversal
        if not neighbors:
            break  # no way forward
        prev = current
        current = random.choice(neighbors)
        path.append(current)
    return path

# kennel per building 
main_kennel = [1, 164, 214, 10, 54, 231, 18, 202, 246, 40, 60, 190, 239, 358, 63, 13, 115, 245, 125, 282]
new_kennel = [340, 230, 265, 133, 119, 135, 3, 139, 330, 221, 291, 281, 11, 99, 218, 348, 295, 141, 188, 228, 216, 19, 105, 182, 49, 148, 170, 39, 81, 9, 94, 173, 325, 177]
seren_kennel = [305, 349, 103, 318, 21, 59, 267, 307, 344, 285, 144, 107, 203, 236]
cwtch_kennel = [194, 298, 302, 273, 97, 24]    # dogs having their own paddock
all_kennels = main_kennel + new_kennel + seren_kennel + cwtch_kennel

# Paddocks per building / special
main_paddock = [313, 253, 38, 284, 74, 93]
seren_paddock = [130, 232, 51]
new_paddock = [127, 303, 308, 276, 102, 184, 126, 7, 156, 32]
special_paddocks = {                                           # dogs having their own paddock
    194:[137], 298:[86], 302:[222], 273:[33],
    97:[152], 24:[17], 348:[171], 295:[300]
}

# Training rooms
main_training = [304, 283, 55]
seren_cwtch_training = [275, 87, 332, 186]
new_training = [336, 339]

# Build dog→paddock map
dog_to_paddock = {}
for d in all_kennels:
    if d in special_paddocks:
        dog_to_paddock[d] = special_paddocks[d]
    elif d in main_kennel:
        dog_to_paddock[d] = main_paddock
    elif d in seren_kennel:
        dog_to_paddock[d] = seren_paddock
    else:
        dog_to_paddock[d] = new_paddock

# Build dog→training map
dog_to_training = {}
for d in all_kennels:
    if d in main_kennel:
        dog_to_training[d] = main_training
    elif d in seren_kennel or d in cwtch_kennel:
        dog_to_training[d] = seren_cwtch_training
    else:
        dog_to_training[d] = new_training

# Walking zones
teletubby = set()
walk = set()
for i, row in gdf.iterrows():
    if 'zone' in row and row['zone'] == 'teletubby':
        coords = list(row.geometry.coords)
        for pt in coords:
            teletubby.add(coord_to_index[pt])
    elif 'zone' in row and row['zone'] == 'walk':
        coords = list(row.geometry.coords)
        for pt in coords:
            walk.add(coord_to_index[pt])

teletubby = list(teletubby)
walk = list(walk)
long_walk = [257, 206, 44, 207, 69, 247, 343, 277, 204, 311, 258, 196, 132, 180, 123, 96, 256, 84, 211, 167, 338, 80, 165, 235, 106, 175, 192, 226, 35, 37, 158, 22, 129] 
walk_subgraph = G.subgraph(walk).copy()
teletubby_subgraph = G.subgraph(teletubby).copy()
long_subgraph = G.subgraph(long_walk).copy()

# --- Setup staff groups across buildings ---
# Main: 4 staff × 5 dogs
staff_dogs = [main_kennel[i * 5 : (i + 1) * 5] for i in range(4) ]
# Seren: 3 staff × [4,5,5]
start = 0
for sz in [4, 5, 5]:
    staff_dogs.append(seren_kennel[start:start+sz])
    start += sz
# Cwtch: 1 staff
staff_dogs.append(cwtch_kennel) 
# New: 6 staff × [5,5,6,6,5,7]
start = 0
for sz in [5, 5, 6, 6, 5, 7]:
    staff_dogs.append(new_kennel[start:start+sz])
    start += sz

num_staff = len(staff_dogs) 


# Initialize states
occupancy = defaultdict(lambda: defaultdict(set)) # Occupancy dict: time_step -> node -> set of ('dog', dog_id) or ('staff', staff_id)
dog_health = {d: 0 for d in all_kennels}
node_contamination = {i: 0 for i in range(len(nodes))}
staff_health = {i: 0 for i in range(num_staff)} 

# Introduce patient zero in the new building
patient_zero = random.choice(new_kennel)
dog_health[patient_zero] = 1
print(f"Patient zero is dog at node {patient_zero}")

# --- Measure simulation time ---
start_time = time.time()

round_start_times = []

# Daily simulation
for day in range(90):

    time_step = day * 24 * 60 * 60  # starting time for each day (in seconds)

    current_time = time_step
    paddock_available_time = {p: current_time for p in main_paddock + seren_paddock + new_paddock + sum(special_paddocks.values(), [])}
    training_room_available_time = {r: current_time for r in main_training + seren_cwtch_training + new_training}
    staff_available_time = [current_time] * num_staff

    for round_num in range(4):        # 4 exercises per day

        round_start_times.append(current_time)

        print(f"\n===== Day {day}, Round {round_num} =====")

        moved_dogs = set()
        staff_location = random.choices(range(len(nodes)), k=num_staff)

        while len(moved_dogs) < len(all_kennels):
            if all(staff_available_time[staff_id] > current_time for staff_id in range(num_staff)):
                current_time = min(staff_available_time)
            
            else:
                available_staff = [i for i in range(num_staff) if staff_available_time[i] <= current_time]
                random.shuffle(available_staff)     # randomly choosing staffs
                staff_moved_dog = False
                for staff_id in available_staff:
                    available_dogs = list(set(staff_dogs[staff_id]) - moved_dogs)
                    if not available_dogs:
                        continue
       
                    start = random.choice(available_dogs)
                    activity_type = random.choices(["paddock", "walk", "training"], weights=[0.3, 0.5, 0.2])[0]
                    try:
                        # staff to kennel
                        staff_to_kennel = nx.shortest_path(G, source=staff_location[staff_id], target=start, weight='weight')
                        staff_to_kennel_time = (len(staff_to_kennel) - 1) * 5
                    except nx.NetworkXNoPath:
                        continue
                    
                    # Staff walking alone to kennel
                    for i, node in enumerate(staff_to_kennel[:-1]):  # skip the last node (kennel)
                        t = current_time + i * 5
                        occupancy[t][node].add(("staff", staff_id))

                    if activity_type == "paddock":
                        available_paddocks = [p for p in dog_to_paddock[start] if paddock_available_time[p] <= current_time]
                        if not available_paddocks:
                            continue
                        end = random.choice(available_paddocks)
                        try:
                            to_paddock = nx.shortest_path(G, source=start, target=end, weight='weight')
                            back_path = nx.shortest_path(G, source=end, target=start, weight='weight')
                            full_path = to_paddock + back_path[1:]
                        except nx.NetworkXNoPath:
                            continue

                        # Staff and dog moving to paddock and back
                        round_time = current_time + staff_to_kennel_time 
                        paddock_available_time[end] = round_time + (len(to_paddock) - 1) * 5 + 900
                        for node in full_path:
                            occupancy[round_time][node].add(("staff", staff_id))
                            occupancy[round_time][node].add(("dog", start))
                            if node == end:
                                for i in range(180):  # 900 seconds / 5s = 180 steps
                                    round_time += 5
                                    occupancy[round_time][node].add(("dog", start))
                                    occupancy[round_time][node].add(("staff", staff_id))
                        
                            round_time += 5

                    elif activity_type == "walk":
                        walk_area = random.choices(["long", "walk", "teletubby"], weights=[0.2, 0.6, 0.2])[0]
                        if walk_area == "long":
                            zone_nodes = long_walk
                            zone_subgraph = long_subgraph
                        elif walk_area == "walk":
                            zone_nodes = walk
                            zone_subgraph = walk_subgraph
                        else:
                            zone_nodes = teletubby
                            zone_subgraph = teletubby_subgraph

                        shortest_path = None    # set shortest path from kennel to the choosing walking area
                        min_length = float('inf')
                        for node in zone_nodes:
                            try:
                                path = nx.shortest_path(G, source=start, target=node, weight='weight')
                                if len(path) < min_length:
                                    shortest_path = path
                                    min_length = len(path)
                            except nx.NetworkXNoPath:
                                continue
                        if shortest_path is None:
                            continue  # no accessible entry
                        
                        to_zone_start = shortest_path
                        to_zone_time = (len(to_zone_start) - 1) * 5

                        # Record dog and staff walking to the zone
                        round_time = current_time + staff_to_kennel_time 
                        for node in to_zone_start[:-1]:
                            occupancy[round_time][node].add(("staff", staff_id))
                            occupancy[round_time][node].add(("dog", start))
                            round_time += 5

                        # Walk inside zone
                        zone_entry = to_zone_start[-1]
                        walk_path = generate_random_walk(zone_subgraph, zone_entry)

                        for node in walk_path:
                            occupancy[round_time][node].add(("staff", staff_id))
                            occupancy[round_time][node].add(("dog", start))
                            round_time += 5

                        # Return to kennel from last walk node
                        walk_end = walk_path[-1]
                        try:
                            return_path = nx.shortest_path(G, source=walk_end, target=start, weight='weight')
                        except nx.NetworkXNoPath:
                            continue

                        for node in return_path[1:]:  # skip current node
                            occupancy[round_time][node].add(("staff", staff_id))
                            occupancy[round_time][node].add(("dog", start))
                            round_time += 5
                    
                    else:
                        available_rooms = [r for r in dog_to_training[start] if training_room_available_time[r] <= current_time]
                        if not available_rooms:
                            continue
                        room = random.choice(available_rooms)

                        try:
                            to_room = nx.shortest_path(G, source=start, target=room, weight='weight')
                            back_path = nx.shortest_path(G, source=room, target=start, weight='weight')
                        except nx.NetworkXNoPath:
                            continue

                        full_path = to_room + back_path[1:]
                        round_time = current_time + staff_to_kennel_time
                        training_room_available_time[room] = round_time + (len(to_room) - 1) * 5 + 900

                        for node in full_path:
                            occupancy[round_time][node].add(("staff", staff_id))
                            occupancy[round_time][node].add(("dog", start))
                            if node == room:
                                for i in range(180):  # 15 min training
                                    round_time += 5
                                    occupancy[round_time][node].add(("staff", staff_id))
                                    occupancy[round_time][node].add(("dog", start))
                
                            round_time += 5

                    staff_moved_dog = True
                    staff_available_time[staff_id] = round_time - 5
                    moved_dogs.add(start)
                    staff_location[staff_id] = start
                    print(f"Staff {staff_id} moving dog {start} for activity: {activity_type}")
                    print(f"Route to start: {staff_to_kennel}")
                    print(f"Activity route: {full_path if activity_type != 'walk' else walk_path}") 
                
                if not staff_moved_dog:
                    # All available staff either had no dogs or their moves failed
                    future_times = [t for t in staff_available_time if t > current_time]
                    if future_times:
                        current_time = min(future_times)
                    else:
                        print("[DEADLOCK] No future staff availability.")
                        break
                                     
#------------------check---------------------------------------------------------------------------------------------------

        if moved_dogs == set(all_kennels):
            print("[CHECK] All dogs were exercised once this round.")
        else:
            missing = set(all_kennels) - moved_dogs
            print(f"[WARNING] Some dogs were not exercised: {missing}")
            sys.exit("[ABORTING] Exiting entire simulation.")
#---------------check---------------------------------------------------------------------------------------------------
        current_time = max(staff_available_time)


# --- End of simulation ---
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Simulation completed in {elapsed_time:.2f} seconds.")

def to_regular_dict(d):
    if isinstance(d, defaultdict):
        d = {k: to_regular_dict(v) for k, v in d.items()}
    return d

occupancy_regular = to_regular_dict(occupancy)

base_dir = os.path.dirname(__file__)
out_path = os.path.join(base_dir, "data", "full_simulation.pkl")
with open(out_path, "wb") as f:
    pickle.dump({
        "occupancy": occupancy_regular,
        "staff_dogs": staff_dogs,
        "staff_available_time": staff_available_time,
        "paddock_available_time": paddock_available_time,
        "training_room_available_time": training_room_available_time,
        "dog_health": dog_health,
        "round_start_times": round_start_times,
        "node_contamination": node_contamination,
        "staff_health": staff_health,
        "graph": G,
        "coord_to_index": coord_to_index
    }, f)

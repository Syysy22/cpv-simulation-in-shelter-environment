# CPV Simulation in a Shelter Environment

This repository accompanies my MSc Data Science dissertation.  
It contains the code and input data used to simulate canine parvovirus (CPV) transmission within a rehoming centre and to evaluate different intervention strategies.

---

## Repository Structure

- **data/**  
  Contains input data files:  
  - `paths.*` — shapefile of the rehoming centre layout.  

- **movement_simulation.py**  
  Generates dog and staff movement data and saves it as `data/full_simulation.pkl` (not stored on GitHub due to size; must be created locally).

- **infection_baseline.py**  
  Runs the infection model using the movement data without interventions under different scenarios (default is low transmission case).

- **experiment_1&2&3/**  
  Scripts and plotting code for:  
  1. Intensifying environmental sanitation  
  2. Intensifying staff hygiene (default intervention and under low transmission rate)
  3. Combined environment + staff interventions  
  Results are saved into `data/cpv_env_staff_combo.pkl` and can be plotted using the scripts in this folder.

- **experiment_4/**  
  Scripts and plotting code for vaccination strategies (default is low transmission case).
  Runs scenarios with different vaccination coverages (0%, 50%, 75%).  
  Results are saved into `data/cpv_vax.pkl`.

- **README.md**  
  This file (project description and usage instructions).

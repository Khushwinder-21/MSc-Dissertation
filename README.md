# Patient Treatment List (PTL) Consolidation Project

This repository contains code and data for modelling the impact of consolidating multiple Patient Treatment Lists (PTLs) across NHS and independent providers. The project combines queueing theory, discrete-event simulation, and reinforcement learning to evaluate waiting time reductions and utilisation improvements.

---

## Repository Contents

### Core Models
- **`MGC_queue.py`**  
  Implements **M/G/c queueing analysis** with backlog adjustments and variability corrections.  
  - Models separate provider queues and consolidated queues.  
  - Uses Erlang-C formulas with a correction for non-exponential service times.  
  - Adjusts service rates and backlog levels to replicate observed 20–80 week waiting times.

- **`DES_simulation.py`**  
  A **Discrete-Event Simulation (DES)** comparing separate vs. consolidated PTLs.  
  - Uses `simpy.PriorityResource` to simulate theatres with priority scheduling (Urgent > Two-Week Wait > Routine).  
  - Models realistic constraints:  
    - 40-week NHS and 26-week Independent synthetic backlogs.  
    - 25% transfer cap for routine patients.  
    - Theatre utilisation capped at 95%.  
    - Cancellations and rescheduling incorporated.  
  - Provides scenario outputs: mean wait times, 90th percentile waits, utilisation, and system throughput.

## Reinforcement Learning Implementations

- **Cross-Entropy Method (CEM)**  
  Adapted from [jerrylin1121/cross_entropy_method](https://github.com/jerrylin1121/cross_entropy_method).  
  - Modified to sample routing policies across multiple providers.  
  - Evaluated policies via simulation, using waiting time as the reward signal.

- **Deep Q-Network (DQN)**  
  Based on [Denny Britz’s RL implementations](https://github.com/dennybritz/reinforcement-learning).  
  - Adapted to model **healthcare queueing** rather than OpenAI Gym tasks.  
  - Environment extended to:  
    - Represent multiple providers with different capacities.  
    - Incorporate initial backlog states (40 weeks NHS, 26 weeks Independent) based on DES calibration.  
  - Reward structure designed around reducing waiting times while respecting transfer caps and priorities.

---

### Data
- **`old_data.csv`** — Anonymized  patient referral records for Cataract provided by ICB Cambridgeshire and Peterborough for past 3 FY
- **`new_data_ref_dates.csv`** — Previous FY cataract surgery data with priority and waiting time.
- **`provider_capacity_data.csv`** — Provider-level capacity information calculated (throughput, operating days, number of theatres).  

### Project Documentation
- **`2630818.pdf`** — Dissertation report: *Optimising Healthcare Resource Allocation Through PTL Consolidation*. Includes methodology, results, and NHS policy context.


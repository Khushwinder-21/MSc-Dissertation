import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

from DES_simulation import (
    # data loading + helpers
    load_patients, load_capacity, infer_provider_weekly_arrivals,
    calibrate_arrivals_per_provider_to_util, build_provider_configs,
    build_synthetic_backlog, load_backlog,
    # types/constants used in DES
    PRIORITY_RANK, NHS_PROVIDERS, ICB_PROVIDERS,
    # sim parameters used for consistency
    RANDOM_SEED, WARMUP_WEEKS, RUN_WEEKS, TIME_HORIZON,
    SERVICE_TIME_CV, SERVICE_RATE_MULTIPLIER,
    MAX_OBS_UTIL, MAX_TRANSFER_SHARE, ELIGIBLE_TO_TRANSFER,
    CANCEL_PROB, RESCHED_MEAN_WEEKS
)

import simpy
# utility functions same as in DES module
def lognormal_params_from_mean_cv(mean, cv):
    if cv <= 0:
        return np.log(max(mean, 1e-9)), 1e-6
    sigma2 = np.log(1 + cv**2)
    mu = np.log(max(mean, 1e-9)) - 0.5 * sigma2
    sigma = np.sqrt(sigma2)
    return mu, sigma

def wait_until_provider_open(env, now):
    wk_start = math.floor(now)
    x = now - wk_start  # [0,1)
    if x <= MAX_OBS_UTIL:
        return None
    return env.timeout((wk_start + 1.0) - now)

@dataclass
class ProviderConfig:
    name: str
    theatres: int
    weekly_capacity: float  # patients/week total (all theatres)

    @property
    def mu_per_server(self):
        return (self.weekly_capacity * SERVICE_RATE_MULTIPLIER) / max(self.theatres, 1)

    @property
    def mean_service_time(self):
        rate = self.mu_per_server
        return 1.0 / max(rate, 1e-9)

class Monitors:
    def __init__(self, providers, provider_cfg):
        self.waits = []          # (provider, wait_time, prio, arrive_t)
        self.system_times = []   # (provider, sys_time, arrive_t)
        self.active_servers = {p: 0 for p in providers}
        self.last_t = {p: 0.0 for p in providers}
        self.busy_integral = {p: 0.0 for p in providers}
        self.cap = {p: provider_cfg[p].theatres for p in providers}
        self.warmup = WARMUP_WEEKS

    def accumulate(self, provider, env):
        t0 = self.last_t[provider]
        t1 = env.now
        if t1 <= self.warmup:
            self.last_t[provider] = t1
            return
        start = max(t0, self.warmup)
        dt = t1 - start
        if dt > 0:
            self.busy_integral[provider] += self.active_servers[provider] * dt
            self.last_t[provider] = t1

    def start_service(self, provider, env):
        self.accumulate(provider, env)
        self.active_servers[provider] += 1

    def end_service(self, provider, env):
        self.accumulate(provider, env)
        self.active_servers[provider] = max(0, self.active_servers[provider] - 1)

    def utilisation(self, provider):
        post_warmup = max(TIME_HORIZON - WARMUP_WEEKS, 1e-9)
        denom = max(self.cap[provider] * post_warmup, 1e-9)
        u = self.busy_integral[provider] / denom
        return float(min(max(u, 0.0), 1.0))

def draw_priority(mix={1:0.7,2:0.1,3:0.2}, return_code=False):
    labels = [1,2,3]
    probs = np.array([mix.get(1,0.7), mix.get(2,0.1), mix.get(3,0.2)], dtype=float)
    probs = probs / probs.sum()
    code = int(np.random.choice(labels, p=probs))
    if return_code:
        return PRIORITY_RANK[code], code
    return PRIORITY_RANK[code]

def patient(env, pid, provider_name, prio, res_map, prov_cfg, monitors):
    gate = wait_until_provider_open(env, env.now)
    if gate is not None:
        yield gate

    arrive_t = env.now
    res = res_map[provider_name]

    req = res.request(priority=prio)
    yield req

    gate2 = wait_until_provider_open(env, env.now)
    if gate2 is not None:
        res.release(req)
        yield gate2
        req = res.request(priority=prio)
        yield req

    wait_t = env.now - arrive_t
    if arrive_t >= WARMUP_WEEKS:
        monitors.waits.append((provider_name, wait_t, prio, arrive_t))

    cancel_p = CANCEL_PROB.get(prio, CANCEL_PROB[2])
    if np.random.rand() < cancel_p:
        res.release(req)
        resched_delay = float(np.random.exponential(RESCHED_MEAN_WEEKS))
        def rescheduled():
            yield env.timeout(resched_delay)
            env.process(patient(env, f"{pid}-RS", provider_name, prio, res_map, prov_cfg, monitors))
        env.process(rescheduled())
        return

    monitors.start_service(provider_name, env)

    mean_st = prov_cfg[provider_name].mean_service_time
    mu, sigma = lognormal_params_from_mean_cv(mean_st, SERVICE_TIME_CV)
    st = float(np.random.lognormal(mean=mu, sigma=sigma))
    yield env.timeout(st)

    monitors.end_service(provider_name, env)
    if arrive_t >= WARMUP_WEEKS:
        monitors.system_times.append((provider_name, env.now - arrive_t, arrive_t))
    res.release(req)

# policy-driven consolidated simulation
def run_consolidated_policy(rep_seed: int,
                            providers_cfg: Dict[str, ProviderConfig],
                            weekly_arrivals_by_home: Dict[str, float],
                            backlog_map: Dict[str, int],
                            prio_mix: Dict[int,float],
                            routing_matrix: pd.DataFrame,
                            enforce_nhs_all_hrgs=True) -> Dict[str, float]:
    np.random.seed(rep_seed)
    env = simpy.Environment()
    res_map = {name: simpy.PriorityResource(env, capacity=cfg.theatres) for name, cfg in providers_cfg.items()}
    monitors = Monitors(list(providers_cfg.keys()), providers_cfg)

    providers = list(providers_cfg.keys())

    # normalising routing rows to <= MAX_TRANSFER_SHARE and >= 0
    R = routing_matrix.reindex(index=providers, columns=providers).fillna(0.0).clip(lower=0.0)
    row_sums = R.sum(axis=1).replace(0, 1.0)
    scale = np.minimum(1.0, MAX_TRANSFER_SHARE / row_sums)
    R = (R.T * scale.values).T  # scaling rows

    #keeping small self-routing to preserve home identity & NHS teaching
    eps = 1e-6
    for h in providers:
        stay = max(0.0, 1.0 - R.loc[h].sum())
        R.loc[h, h] += stay
    if enforce_nhs_all_hrgs:
        # making sure every NHS node receives some routine flow from system
        nhs_nodes = [p for p in providers if p in NHS_PROVIDERS]
        if nhs_nodes:
            # adding small epsilon value to each NHS dest from each home
            for h in providers:
                take = min(eps * len(nhs_nodes), R.loc[h, h] * 0.5)
                if take > 0:
                    R.loc[h, h] -= take
                    add = take / len(nhs_nodes)
                    for d in nhs_nodes:
                        R.loc[h, d] += add

    arrivals_count = {p: 0 for p in providers}
    transfers_used = {p: 0 for p in providers}

    # base provider arrival 
    def home_arrival_process(home_provider, weekly_lambda):
        while True:
            lam = max(weekly_lambda, 1e-9)
            iat = np.random.exponential(scale=1.0/lam)
            yield env.timeout(iat)
            prio_rank, prio_code = draw_priority(mix=prio_mix, return_code=True)
            arrivals_count[home_provider] += 1

            # default destination is original provider
            dest = home_provider

            # only Routine priority is eligible to move as in DES
            if ELIGIBLE_TO_TRANSFER.get(prio_code, False):
                # enforce 25% cap per provider
                cap_remaining = (transfers_used[home_provider] < MAX_TRANSFER_SHARE * max(arrivals_count[home_provider], 1))
                if cap_remaining:
                    # draw destination based on routing matrix row
                    probs = R.loc[home_provider].values.astype(float)
                    probs = probs / max(probs.sum(), 1e-12)
                    dest_idx = np.random.choice(len(providers), p=probs)
                    dest = providers[dest_idx]
                    if dest != home_provider:
                        transfers_used[home_provider] += 1
                else:
                    dest = home_provider

            env.process(patient(env,
                                f"POL-{home_provider}->{dest}-{env.now:.3f}",
                                dest, prio_rank, res_map, providers_cfg, monitors))

    # launching arrival processes
    for home, lam in weekly_arrivals_by_home.items():
        env.process(home_arrival_process(home, lam))

    # seeding backlog at home providers (as in baseline)
    for name, size in backlog_map.items():
        for _ in range(int(size)):
            prio_rank = draw_priority(mix=prio_mix, return_code=False)
            env.process(patient(env, f"BL-{name}-{env.now:.6f}", name, prio_rank, res_map, providers_cfg, monitors))

    env.run(until=TIME_HORIZON)

    waits = [w for (p, w, pr, at) in monitors.waits if w is not None and (w >= 0)]
    sys_times = [s for (p, s, at) in monitors.system_times if s is not None]
    util = {p: monitors.utilisation(p) for p in providers}

    moved = sum(transfers_used.values())
    total = sum(arrivals_count.values()) or 1

    return {
        "mean_wait": float(np.mean(waits)) if waits else 0.0,
        "p90_wait": float(np.percentile(waits, 90)) if waits else 0.0,
        "mean_system": float(np.mean(sys_times)) if sys_times else 0.0,
        "util": util,
        "moved_share_pct": 100.0 * moved / total
    }

# Cross-Entropy Method (CEM) for policy search
class RoutingRL:
    def __init__(self,
                 patients_csv="new_data_ref_dates.csv",
                 capacity_csv="provider_capacity_data.csv",
                 backlog_csv="backlog_counts.csv", #if empty, will build synthetic 40w/26w
                 n_elite=5, pop=24, iters=25,
                 cost_w=0.15, util_spread_w=0.02, nhs_starve_w=0.5,
                 seed=7):
        np.random.seed(seed)
        self.patients = load_patients(patients_csv)
        self.capacity = load_capacity(capacity_csv)
        self.capacity = self.capacity[self.capacity["provider"].isin(ICB_PROVIDERS)].copy()

        # provider configs
        self.providers_cfg = build_provider_configs(self.capacity)
        self.providers = list(self.providers_cfg.keys())

        # aligning patients to available providers
        self.patients = self.patients[self.patients["provider"].isin(self.providers)].copy()

        # priority mix  (from data)
        prc = getattr(load_patients, "empirical_prio_mix", {1:0.7,2:0.1,3:0.2})
        s = sum(prc.values()) or 1.0
        self.prio_mix = {k: v/s for k,v in prc.items()}

        # weekly arrivals
        weekly_arrivals_raw = infer_provider_weekly_arrivals(self.patients)
        self.weekly_arrivals = calibrate_arrivals_per_provider_to_util(self.providers_cfg, weekly_arrivals_raw)

        # backlog (initial) which is synthetic if not provided
        self.backlog = load_backlog(backlog_csv)
        if not self.backlog:
            self.backlog = build_synthetic_backlog(self.providers_cfg)

        # compute SEPARATE baseline mean wait(avg)
        self.baseline_wait = self.estimate_baseline_wait(n_reps=6)
        print("Providers:", ", ".join(self.providers))
        print(f"Baseline (separate) mean wait: {self.baseline_wait:.2f} weeks")

        # RL params
        self.n_elite = n_elite
        self.pop = pop
        self.iters = iters
        self.cost_w = cost_w
        self.util_spread_w = util_spread_w
        self.nhs_starve_w = nhs_starve_w

        # initialising policy mean/sd in unconstrained space
        n = len(self.providers)
        self.theta_mu = np.zeros((n, n), dtype=float)  # logits
        self.theta_sd = np.ones((n, n), dtype=float) * 1.0

        self.best = None
        self.log_rows = []

    def estimate_baseline_wait(self, n_reps=6):
        from DES_simulation import run_separate
        waits = []
        base = RANDOM_SEED + 123
        for r in range(n_reps):
            out = run_separate(base + 101*r, self.providers_cfg, self.weekly_arrivals, self.backlog)
            waits.append(out["mean_wait"])
        return float(np.mean(waits)) if waits else 0.0

    def sample_policy(self):
        z = self.theta_mu + self.theta_sd * np.random.randn(*self.theta_mu.shape)
        # converting logits to routing matrix
        n = len(self.providers)
        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            # excluding self in softmax to represent move distribution
            logits = np.delete(z[i], i)
            expv = np.exp(logits - logits.max())
            probs = expv / expv.sum()
            # scale to cap
            M_row = probs * MAX_TRANSFER_SHARE
            # reinsert self=0
            row = np.insert(M_row, i, 0.0)
            M[i] = row
        # add stay
        for i in range(n):
            stay = 1.0 - M[i].sum()
            M[i, i] += max(stay, 0.0)
        return pd.DataFrame(M, index=self.providers, columns=self.providers)

    def evaluate_policy(self, routing_df: pd.DataFrame, rep_seed: int = 2024) -> Tuple[float, dict]:
        # run policy‑driven consolidated sim
        out = run_consolidated_policy(rep_seed, self.providers_cfg, self.weekly_arrivals,
                                      self.backlog, self.prio_mix, routing_df,
                                      enforce_nhs_all_hrgs=True)
        mean_wait = out["mean_wait"]
        util = out["util"]
        moved_share = out["moved_share_pct"]

        base = max(self.baseline_wait, 1e-9)
        improvement = 100.0 * (self.baseline_wait - mean_wait) / base

        util_vals = np.array(list(util.values()), dtype=float)
        util_spread_pp = 100.0 * (util_vals.max() - util_vals.min())

        # NHS starvation check
        nhs_utils = [util[p] for p in self.providers if p in NHS_PROVIDERS]
        nhs_starved = int(any(u < 0.05 for u in nhs_utils))  

        #  encourage modest movement
        cost = 0.01 * moved_share 

        # reward = improvement minus penalties
        reward = improvement - (self.cost_w * cost + self.util_spread_w * util_spread_pp + self.nhs_starve_w * nhs_starved)

        info = {
            "improvement_pct": float(improvement),
            "mean_wait_weeks": float(mean_wait),
            "baseline_wait_weeks": float(self.baseline_wait),
            "moved_share_pct": float(moved_share),
            "util_range_pp": float(util_spread_pp),
            "nhs_starved": int(nhs_starved),
            "cost": float(cost),
            "reward": float(reward)
        }
        return reward, info

    def train(self):
        best = None
        for it in range(self.iters):
            batch = []
            for k in range(self.pop):
                policy = self.sample_policy()
                r, info = self.evaluate_policy(policy, rep_seed=RANDOM_SEED + it*997 + k*17)
                batch.append((r, policy, info))

            batch.sort(key=lambda x: x[0], reverse=True)  # higher reward is better
            elites = batch[:self.n_elite]
            elite_rewards = [e[0] for e in elites]
            elite_policies = [e[1].values for e in elites]
            elite_infos = [e[2] for e in elites]

            # update distribution (CEM)
            E = np.stack([np.log(np.maximum(p, 1e-12)) for p in elite_policies], axis=0)  # log to keep smooth update
            self.theta_mu = 0.7*self.theta_mu + 0.3*np.mean(E, axis=0)
            self.theta_sd = 0.7*self.theta_sd + 0.3*np.std(E, axis=0)

            elite = elites[0]
            if (best is None) or (elite[0] > best[0]):
                best = elite

            # logging
            elite_info = elite[2]
            print(f"Iter {it:02d} | EliteReward {elite[0]:.2f} | BestImprovement {best[2]['improvement_pct']:.2f}% | Cost {elite_info['cost']:.2f}")
            self.log_rows.append({
                "iter": it,
                "elite_reward": elite[0],
                "elite_improvement_pct": elite_info["improvement_pct"],
                "elite_cost": elite_info["cost"]
            })


        best_reward, best_policy_df, best_info = best
        np.savez("rl_policy.npz", routing_matrix=best_policy_df.values,
                 providers=np.array(self.providers, dtype=object))
        best_policy_df.to_csv("rl_routing_shares_routine_moved.csv", index=True)
        print("Best policy report:", best_info)
        return best_policy_df, best_info


if __name__ == "__main__":
    rl = RoutingRL(
        n_elite=6, pop=28, iters=25,
        cost_w=0.20, util_spread_w=0.02, nhs_starve_w=0.8, 
        seed=13
    )
    best_policy, report = rl.train()

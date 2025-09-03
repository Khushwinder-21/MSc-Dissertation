import simpy
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
import math
#config for the des
RANDOM_SEED = 42
N_REPS = 10
WARMUP_WEEKS = 8         # exclude metrics before this time
RUN_WEEKS = 52           # data collection window
TIME_HORIZON = (WARMUP_WEEKS + RUN_WEEKS)  # in weeks

# Date formats to try (sample shows 12/18/2024)
DATE_FORMATS = ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"]

# Input files
PATIENTS_CSV = "new_data_ref_dates.csv"
CAPACITY_CSV = "provider_capacity_data.csv"
BACKLOG_CSV = "backlog_counts.csv"  # If we get it then we can use it for now we have added the flow which uses synthetic data

# providers considered
ICB_PROVIDERS = {
    "ANGLIA COMMUNITY EYE SERVICE LTD",
    "SPAMEDICA PETERBOROUGH",
    "SPAMEDICA BEDFORD",
    "FITZWILLIAM HOSPITAL",
    "NORTH WEST ANGLIA NHS FOUNDATION TRUST",
    "CAMBRIDGE UNIVERSITY HOSPITALS NHS FOUNDATION TRUST",
}

# re defining the provider classification
NHS_PROVIDERS = {
    "NORTH WEST ANGLIA NHS FOUNDATION TRUST",
    "CAMBRIDGE UNIVERSITY HOSPITALS NHS FOUNDATION TRUST",
}
INDEPENDENT_PROVIDERS = ICB_PROVIDERS - NHS_PROVIDERS

# assumed backlog  (weeks)
ASSUMED_INITIAL_WAIT_NHS_WEEKS = 40
ASSUMED_INITIAL_WAIT_INDEP_WEEKS = 26

# assumed current utilisation
ASSUMED_UTIL_NHS   = 0.93
ASSUMED_UTIL_INDEP = 0.88
MAX_TARGET_UTIL = 0.975   # safety clip

# Consolidation realism PARAMETERS
ROUTINE_CODE = 1  # source data convention
ELIGIBLE_TO_TRANSFER = {1: True, 2: False, 3: False}  # Routine only
MAX_TRANSFER_SHARE = 0.25  # ≤ 25% of arrivals per home provider can be redirected

# Columns in patient CSV to handle descripancies in naming of the columns
PROVIDER_COL_CANDIDATES = ["Provider", "provider", "TrustName", "trust"]
DATE_COL_CANDIDATES = [
    "Start_clock_date", "End_clock_date",
    "ReferralDate", "Referral_Date", "Date", "Referral Start Date", "referral_date"
]
PRIORITY_COL_CANDIDATES = ["Priority", "priority", "priority_code", "PriorityCode"]

# 1 - Routine, 2 - Urgent, 3 - Two Week Wait
PRIORITY_RANK = {1: 2, 2: 0, 3: 1}

# Service time model (weeks) per case; we derive rate from capacity and convert to mean service time
SERVICE_TIME_CV = 0.6  # M/G/c variability (lognormal approx)
SERVICE_RATE_MULTIPLIER = 1.0 

#utilisation cap & reliability parameters
MAX_OBS_UTIL = 0.95           # max capacity limit
CLOSES_LAST_FRACTION = 1.0 - MAX_OBS_UTIL  # fraction of each week when theatres are closed

# Cancellation / rescheduling
CANCEL_PROB = {
    0: 0.03,   # Urgent
    1: 0.05,   # TWW
    2: 0.07,   # Routine
}
RESCHED_MEAN_WEEKS = 1.5      # average delay before the patient returns after a cancellation/no-show

#helping functions
def pick_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"None of {candidates} found in dataframe columns: {list(df.columns)}")


def parse_dates_series(s: pd.Series) -> pd.Series:
    s_out = None
    for fmt in DATE_FORMATS:
        try:
            s_out = pd.to_datetime(s, format=fmt, errors="raise")
            return s_out
        except Exception:
            continue
    return pd.to_datetime(s, errors="coerce")


def load_patients(path: str):
    df = pd.read_csv(path)
    prov_col = pick_column(df, PROVIDER_COL_CANDIDATES)
    try:
        date_col = pick_column(df, ["Start_clock_date"] + DATE_COL_CANDIDATES)
    except KeyError:
        date_col = pick_column(df, DATE_COL_CANDIDATES)
    prio_col = pick_column(df, PRIORITY_COL_CANDIDATES)

    # Parse dates
    df[date_col] = parse_dates_series(df[date_col])
    df = df.dropna(subset=[date_col])

    # Map priority codes to ranks AND keep original code for transfer eligibility
    df["prio_code"] = df[prio_col].astype(int)
    df["prio_rank"] = df["prio_code"].map(PRIORITY_RANK).fillna(2).astype(int)

    out = df[[prov_col, date_col, "prio_rank", "prio_code"]].rename(
        columns={prov_col: "provider", date_col: "ref_date", "prio_rank": "prio", "prio_code": "prio_code"}
    )

    # basic priority mix used for generating arrivals
    prio_counts = out["prio_code"].value_counts(normalize=True)
    mix = {1: 0.7, 2: 0.1, 3: 0.2}
    for k in [1, 2, 3]:
        if k in prio_counts.index:
            mix[k] = float(prio_counts[k])
    s = sum(mix.values()) or 1.0
    load_patients.empirical_prio_mix = {k: v / s for k, v in mix.items()}

    return out


def load_capacity(path: str):
    cap = pd.read_csv(path)
    if "provider" not in cap.columns:
        for c in ["Provider", "TrustName", "trust"]:
            if c in cap.columns:
                cap = cap.rename(columns={c: "provider"})
                break
    cap = cap.copy()
    if "theatres" not in cap.columns:
        cap["theatres"] = 1
    if "days_per_week" not in cap.columns:
        cap["days_per_week"] = 5
    if "current_daily_throughput" not in cap.columns:
        cap["current_daily_throughput"] = 1.0
    return cap


def load_backlog(path: str):
    try:
        b = pd.read_csv(path)
        if "backlog_size" not in b.columns or "provider" not in b.columns:
            raise ValueError("backlog_counts.csv must have columns [provider, backlog_size]")
        return dict(zip(b["provider"], b["backlog_size"].astype(int)))
    except Exception:
        return {}


def year_week(dt: pd.Timestamp):
    iso = dt.isocalendar()
    return (iso.year, iso.week)


def infer_provider_weekly_arrivals(pat_df: pd.DataFrame):
    # group by provider and ISO week to get weekly counts
    pat_df = pat_df.copy()
    pat_df["yearweek"] = pat_df["ref_date"].apply(year_week)
    wk = pat_df.groupby(["provider", "yearweek"]).size().reset_index(name="n")
    arr = wk.groupby("provider")["n"].mean().to_dict()
    for k in arr:
        if arr[k] <= 0:
            arr[k] = 0.1
    return arr


def lognormal_params_from_mean_cv(mean, cv):
    if cv <= 0:
        return np.log(max(mean, 1e-9)), 1e-6
    sigma2 = np.log(1 + cv**2)
    mu = np.log(max(mean, 1e-9)) - 0.5 * sigma2
    sigma = np.sqrt(sigma2)
    return mu, sigma


def wait_until_provider_open(env, now):
    """
    this function enforces that theatres are 'closed' for the last (1 - MAX_OBS_UTIL) fraction of every week.
    If called during the closed window, this gives until the next week starts.
    """
    wk_start = math.floor(now)
    x = now - wk_start  # position within the current week [0,1)
    if x <= MAX_OBS_UTIL:
        return None  # already open
    # wait till next week
    dt = (wk_start + 1.0) - now
    return env.timeout(dt)

#des components
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
        # mean time a theatre is occupied by one case (in weeks)
        rate = self.mu_per_server
        return 1.0 / max(rate, 1e-9)


class Monitors:
    def __init__(self, providers, provider_cfg):
        self.waits = []  # (provider, wait_time, prio, arrive_t)
        self.system_times = []  # (provider, sys_time, arrive_t)
        self.active_servers = {p: 0 for p in providers}  # number of busy servers at time t
        self.last_t = {p: 0.0 for p in providers}
        self.busy_integral = {p: 0.0 for p in providers}  # active_servers dt
        self.cap = {p: provider_cfg[p].theatres for p in providers}
        self.warmup = WARMUP_WEEKS

    def accumulate(self, provider, env):
        # accumulate only for time after warm-up
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
        # average fraction of servers busy post-warmup = busy_integral / (cap * post_warmup_time)
        post_warmup = max(TIME_HORIZON - WARMUP_WEEKS, 1e-9)
        denom = max(self.cap[provider] * post_warmup, 1e-9)
        u = self.busy_integral[provider] / denom
        return float(min(max(u, 0.0), 1.0))  # clamp to [0,1]

#patient and arrival processes
def get_empirical_prio_mix():
    return getattr(load_patients, "empirical_prio_mix", {1: 0.7, 2: 0.1, 3: 0.2})

def draw_priority(return_code=False):
    mix = get_empirical_prio_mix()
    labels = [1, 2, 3]
    probs = [mix.get(1, 0.7), mix.get(2, 0.1), mix.get(3, 0.2)]
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()
    code = int(np.random.choice(labels, p=probs))
    if return_code:
        return PRIORITY_RANK[code], code
    return PRIORITY_RANK[code]


def patient(env, pid, provider_name, prio, res_map, prov_cfg, monitors):
    """
    Single patient flow: 
    Wait until provider is 'open' ,
    Join the priority queue, 
    On start: possible cancellation/reschedule,
    Otherwise get service (lognormal),
    Record KPIs (post-warmup).
    """
    # If provider is inside its closed weekly, wait until it re-opens
    gate = wait_until_provider_open(env, env.now)
    if gate is not None:
        yield gate # wait until open

    arrive_t = env.now
    res = res_map[provider_name]

    # Request theatre with priority
    req = res.request(priority=prio)
    yield req # wait in queue

    # If provider moved into the closed tail while we waited, delay actual start until open
    gate2 = wait_until_provider_open(env, env.now)
    if gate2 is not None:
        # release, wait for open, then re-request to preserve fairness
        res.release(req)
        yield gate2
        req = res.request(priority=prio)
        yield req

    # Measure queue wait up to the moment service is about to start
    wait_t = env.now - arrive_t
    F_patient = draw_patient_factor()
    # Dynamic cancellation probability
    P_cancel = P_BASE * (1 + ALPHA * wait_t) * F_patient
    P_cancel = min(P_cancel, 1.0)  # cap at 100%
    if arrive_t >= WARMUP_WEEKS:
        monitors.waits.append((provider_name, wait_t, prio, arrive_t))

    # # CANCELLATION / NO-SHOW at start (patient or provider)
    # cancel_p = CANCEL_PROB.get(prio, CANCEL_PROB[2])
    if np.random.rand() < P_cancel:
        res.release(req)
        # reschedule after an exponential delay
        resched_delay = float(np.random.exponential(RESCHED_MEAN_WEEKS))
        monitors['cancellations'] += 1
        def rescheduled():
            yield env.timeout(resched_delay)
            env.process(patient(env, f"{pid}-RS", provider_name, prio, res_map, prov_cfg, monitors))
        env.process(rescheduled())
        return 

    #begin service so the server is busy now
    monitors.start_service(provider_name, env)

    # derieve service time based on provider mean & variability
    mean_st = prov_cfg[provider_name].mean_service_time
    mu, sigma = lognormal_params_from_mean_cv(mean_st, SERVICE_TIME_CV) #Healthcare times are positively skewed
    st = float(np.random.lognormal(mean=mu, sigma=sigma))
    yield env.timeout(st)

    # finish the service
    monitors.end_service(provider_name, env)
    if arrive_t >= WARMUP_WEEKS:
        monitors.system_times.append((provider_name, env.now - arrive_t, arrive_t))
    res.release(req)

P_BASE = 0.05   # 5% baseline cancellation probability
ALPHA = 0.01    # sensitivity to waiting time (per week)

def draw_patient_factor():
    """Patient-specific cancellation modifier."""
    return np.random.choice([0.8, 1.0, 1.2], p=[0.2, 0.6, 0.2])

def f_day(t):
    """Day-of-week factor (t in days)."""
    day = int(t) % 7
    if day in [0, 1]:  # Monday, Tuesday
        return 1.2
    elif day in [5, 6]:  # Saturday, Sunday
        return 0.8
    else:
        return 1.0

def f_week(t):
    """Week-of-year factor (t in weeks)."""
    week = int(t) % 52
    if 48 <= week <= 52:  # Christmas dip
        return 0.7
    elif 1 <= week <= 8:  # January–February surge
        return 1.3
    else:
        return 1.0

def f_season(t):
    """Seasonal cycle factor (t in weeks)."""
    # Simple sinusoidal variation over the year
    import math
    return 1.0 + 0.2 * math.sin(2 * math.pi * (t % 52) / 52)



def arrival_process(env, provider_name, weekly_lambda, res_map, prov_cfg, monitors):
    # Poisson arrivals with exponential inter-arrival times (in weeks)
    while True:
        t = env.now / 7.0  # if env.now is in days
        lam_t = weekly_lambda * f_day(env.now) * f_week(t) * f_season(t)
        lam = max(lam_t, 1e-9)
        iat = np.random.exponential(scale=1.0/lam)
        yield env.timeout(iat)
        prio = draw_priority()
        env.process(patient(env, f"{provider_name}-{env.now:.3f}", provider_name, prio, res_map, prov_cfg, monitors))


def seed_backlog(env, provider_name, backlog_n, res_map, prov_cfg, monitors):
    # seed backlog at time 0
    for i in range(int(backlog_n)):
        prio = draw_priority()
        env.process(patient(env, f"BL-{provider_name}-{i}", provider_name, prio, res_map, prov_cfg, monitors))

#different scenario testing functions

def run_separate(rep_seed, providers_cfg, weekly_arrivals, backlog_map):
    np.random.seed(rep_seed)
    env = simpy.Environment()

    res_map = {name: simpy.PriorityResource(env, capacity=cfg.theatres) for name, cfg in providers_cfg.items()}
    monitors = Monitors(providers_cfg.keys(), providers_cfg)

    # Start arrival processes per provider
    for name, lam in weekly_arrivals.items():
        env.process(arrival_process(env, name, lam, res_map, providers_cfg, monitors))

    # Seed backlogs
    for name, size in backlog_map.items():
        if name in providers_cfg:
            seed_backlog(env, name, size, res_map, providers_cfg, monitors)

    env.run(until=TIME_HORIZON)

    waits = [w for (p, w, pr, at) in monitors.waits if w is not None and (w >= 0)]
    sys_times = [s for (p, s, at) in monitors.system_times if s is not None]
    util = {p: monitors.utilisation(p) for p in providers_cfg.keys()}

    return {
        "mean_wait": float(np.mean(waits)) if waits else 0.0,
        "p90_wait": float(np.percentile(waits, 90)) if waits else 0.0,
        "mean_system": float(np.mean(sys_times)) if sys_times else 0.0,
        "util": util,
        "n_patients": len(sys_times)
    }


def run_consolidated(rep_seed, providers_cfg, weekly_arrivals_by_home, backlog_map, print_transfer_share=True):
    """Consolidated routing with realistic constraints.
    Preserve each patients home provider arrival stream, then allow redirection to the currently best queue.
    """
    np.random.seed(rep_seed)
    env = simpy.Environment()

    res_map = {name: simpy.PriorityResource(env, capacity=cfg.theatres) for name, cfg in providers_cfg.items()}
    monitors = Monitors(providers_cfg.keys(), providers_cfg)

    arrivals_count = {p: 0 for p in providers_cfg.keys()}
    transfers_used = {p: 0 for p in providers_cfg.keys()}

    def home_arrival_process(home_provider, weekly_lambda):
        while True:
            lam = max(weekly_lambda, 1e-9)
            iat = np.random.exponential(scale=1.0/lam)
            yield env.timeout(iat)
            prio_rank, prio_code = draw_priority(return_code=True)
            arrivals_count[home_provider] += 1

            # check if we can transfer this patient based on priority and capacity used so far
            can_transfer = ELIGIBLE_TO_TRANSFER.get(prio_code, False)
            cap_remaining = (transfers_used[home_provider] < MAX_TRANSFER_SHARE * max(arrivals_count[home_provider], 1))

            dest = home_provider
            if can_transfer and cap_remaining:
                # pick destination by heuristic: projected delay = queue length/c + mean service time
                best_p = home_provider
                best_score = 1e9
                for name, res in res_map.items():
                    q_len = len(res.queue)
                    c = providers_cfg[name].theatres
                    score = (q_len / max(c,1)) + providers_cfg[name].mean_service_time
                    if score < best_score:
                        best_score = score
                        best_p = name
                dest = best_p
                if dest != home_provider:
                    transfers_used[home_provider] += 1

            env.process(patient(env, f"CONS-{home_provider}->{dest}-{env.now:.3f}", dest, prio_rank, res_map, providers_cfg, monitors))

    # begin processes
    for home, lam in weekly_arrivals_by_home.items():
        env.process(home_arrival_process(home, lam))

    # put backlogs at home providers
    for name, size in backlog_map.items():
        if name in providers_cfg:
            seed_backlog(env, name, size, res_map, providers_cfg, monitors)

    env.run(until=TIME_HORIZON)

    waits = [w for (p, w, pr, at) in monitors.waits if w is not None and (w >= 0)]
    sys_times = [s for (p, s, at) in monitors.system_times if s is not None]
    util = {p: monitors.utilisation(p) for p in providers_cfg.keys()}

    if print_transfer_share:
        moved = sum(transfers_used.values())
        total = sum(arrivals_count.values()) or 1
        print(f"Transferred share: {100.0*moved/total:.1f}%")

    return {
        "mean_wait": float(np.mean(waits)) if waits else 0.0,
        "p90_wait": float(np.percentile(waits, 90)) if waits else 0.0,
        "mean_system": float(np.mean(sys_times)) if sys_times else 0.0,
        "util": util,
        "n_patients": len(sys_times)
    }

# experiment runner
def build_provider_configs(capacity_df: pd.DataFrame):
    cfg = {}
    for _, r in capacity_df.iterrows():
        name = r.get("provider", r.get("Provider", "Unknown"))
        theatres = int(r.get("theatres", 1))
        weekly_capacity = float(r.get("current_daily_throughput", 1.0)) * float(r.get("days_per_week", 5))
        cfg[name] = ProviderConfig(name=name, theatres=theatres, weekly_capacity=weekly_capacity)
    return cfg


def build_synthetic_backlog(providers_cfg: dict) -> dict:
    backlog = {}
    for name, cfg in providers_cfg.items():
        if name in NHS_PROVIDERS:
            target_weeks = ASSUMED_INITIAL_WAIT_NHS_WEEKS
        else:
            target_weeks = ASSUMED_INITIAL_WAIT_INDEP_WEEKS
        backlog[name] = int(round(target_weeks * max(cfg.weekly_capacity, 1e-6)))
    return backlog


def calibrate_arrivals_per_provider_to_util(providers_cfg: dict, raw_weekly_arrivals: dict) -> dict:
    """Scale each provider arrival rate so their separate baseline offered utilisation hits target."""
    scaled = {}
    for name, cfg in providers_cfg.items():
        lam_raw = float(raw_weekly_arrivals.get(name, 0.1))
        mu = cfg.mu_per_server
        c = cfg.theatres
        if name in NHS_PROVIDERS:
            target = min(ASSUMED_UTIL_NHS, MAX_TARGET_UTIL)
        else:
            target = min(ASSUMED_UTIL_INDEP, MAX_TARGET_UTIL)
        desired_lambda = target * c * mu
        scale = 1.0 if lam_raw <= 0 else desired_lambda / lam_raw
        scaled[name] = max(lam_raw * scale, 1e-6)
    return scaled


def run_experiment():
    patients = load_patients(PATIENTS_CSV)
    capacity = load_capacity(CAPACITY_CSV)

    # Restrict to the top providers; then build config
    capacity = capacity[capacity["provider"].isin(ICB_PROVIDERS)].copy()
    providers_cfg = build_provider_configs(capacity)

    patients = patients[patients["provider"].isin(providers_cfg.keys())].copy()

    # Infer per-provider weekly arrivals from history
    weekly_arrivals_raw = infer_provider_weekly_arrivals(patients)

    # Calibrate arrivals per provider to target offered utilisation for the SEPARATE baseline
    weekly_arrivals = calibrate_arrivals_per_provider_to_util(providers_cfg, weekly_arrivals_raw)

    # Backlog: take data from file but for now we have synthetic data
    backlog_map = load_backlog(BACKLOG_CSV)
    if not backlog_map:
        backlog_map = build_synthetic_backlog(providers_cfg)

    # Offered utilisation table (calibration check)
    offered_util = {
        name: weekly_arrivals[name] / max(providers_cfg[name].theatres * providers_cfg[name].mu_per_server, 1e-9)
        for name in providers_cfg
    }

    results = []
    util_rows = []

    base_seed = RANDOM_SEED
    for rep in range(N_REPS):
        sep = run_separate(base_seed + rep*101, providers_cfg, weekly_arrivals, backlog_map)
        con = run_consolidated(base_seed + 999 + rep*101, providers_cfg, weekly_arrivals, backlog_map, print_transfer_share=True)

        results.append({"scenario": "separate", "mean_wait_weeks": sep["mean_wait"],
                        "p90_wait_weeks": sep["p90_wait"], "mean_system_weeks": sep["mean_system"],
                        "n_patients": sep["n_patients"]})
        results.append({"scenario": "consolidated", "mean_wait_weeks": con["mean_wait"],
                        "p90_wait_weeks": con["p90_wait"], "mean_system_weeks": con["mean_system"],
                        "n_patients": con["n_patients"]})

        for p, u in sep["util"].items():
            util_rows.append({"scenario": "separate", "provider": p, "utilisation": u})
        for p, u in con["util"].items():
            util_rows.append({"scenario": "consolidated", "provider": p, "utilisation": u})

    res_df = pd.DataFrame(results)
    util_df = pd.DataFrame(util_rows)

    summary = res_df.groupby("scenario").agg(
        mean_wait_weeks=("mean_wait_weeks", "mean"),
        p90_wait_weeks=("p90_wait_weeks", "mean"),
        mean_system_weeks=("mean_system_weeks", "mean"),
        n_patients=("n_patients", "sum")
    ).reset_index()

    # % improvement seperate vs consolidated
    try:
        sep_mean = float(summary.loc[summary.scenario=="separate", "mean_wait_weeks"].values[0])
        con_mean = float(summary.loc[summary.scenario=="consolidated", "mean_wait_weeks"].values[0])
        improvement = 100.0 * (sep_mean - con_mean) / max(sep_mean, 1e-9)
    except Exception:
        improvement = np.nan

    print("Summary after ", N_REPS, "replications (post-warmup)")
    print(summary.to_string(index=False))

    # Observed utilisation after warmup
    print("Utilisation by provider (mean):")
    print(util_df.groupby(["scenario", "provider"]).utilisation.mean().reset_index().to_string(index=False))

    # Offered utilisation (from calibrated λ)
    offered_rows = []
    for name, val in offered_util.items():
        offered_rows.append({"provider": name, "offered_util": float(val)})
    offered_df = pd.DataFrame(offered_rows)
    print("Offered utilisation targets (separate baseline):")
    print(offered_df.to_string(index=False))

    if pd.notna(improvement):
        print(f"% improvement in mean wait (consolidated vs separate): {improvement:.2f}%")

if __name__ == "__main__":
    run_experiment()

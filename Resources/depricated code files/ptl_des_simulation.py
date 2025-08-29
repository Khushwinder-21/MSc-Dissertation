"""
Patient Treatment List (PTL) Consolidation Simulation
Discrete Event Simulation for Healthcare Resource Allocation
Optimized for AWS c5.2xlarge (8 vCPUs, 16 GB RAM)
"""

import simpy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== Configuration ==================
class Config:
    """Simulation configuration parameters"""
    # AWS c5.2xlarge optimization
    N_CORES = mp.cpu_count()  # Should be 8 for c5.2xlarge
    CHUNK_SIZE = 100  # Patients per chunk for parallel processing
    
    # Simulation parameters
    SIM_DURATION = 365  # Days
    WARMUP_PERIOD = 30  # Days
    N_REPLICATIONS = 20  # Number of simulation runs
    
    # Current system parameters
    NHS_CURRENT_WAIT = 30  # weeks
    INDEPENDENT_CURRENT_WAIT = 20  # weeks
    NHS_UTILIZATION = 0.875  # 85-90% average
    INDEPENDENT_UTILIZATION = 0.775  # 75-80% average
    
    # Priority weights (2 > 3 > 1)
    PRIORITY_WEIGHTS = {2: 3.0, 3: 2.0, 1: 1.0}
    
    # Seed for reproducibility
    RANDOM_SEED = 42

# ================== Data Classes ==================
@dataclass
class Patient:
    """Patient entity in the simulation"""
    id: str
    hrg: str
    hrg_description: str
    priority: int
    arrival_time: float
    start_clock_date: datetime
    provider: str = None
    activity_type: str = None
    lsoa: str = None
    
    # Tracking metrics
    wait_time: float = 0
    treatment_time: float = 0
    assigned_provider: str = None
    treatment_start: float = None
    treatment_end: float = None
    
    def __lt__(self, other):
        """Priority comparison for queue management"""
        # Higher priority number means higher priority (2 > 3 > 1)
        if self.priority != other.priority:
            return Config.PRIORITY_WEIGHTS[self.priority] > Config.PRIORITY_WEIGHTS[other.priority]
        # If same priority, older patient gets priority (FIFO within priority)
        return self.start_clock_date < other.start_clock_date

@dataclass
class Provider:
    """Healthcare provider entity"""
    name: str
    provider_type: str  # NHS or Independent
    annual_capacity: int
    daily_capacity: float
    current_daily_throughput: float
    procedures_per_hour: float
    weighted_service_time: float
    operating_days: float
    
    # Resources
    theatres: simpy.Resource = None
    
    # Tracking metrics
    patients_treated: int = 0
    total_wait_time: float = 0
    utilization_time: float = 0
    
    def calculate_backlog(self, wait_weeks: int) -> int:
        """Calculate initial backlog based on wait time"""
        return int((wait_weeks / 52) * self.annual_capacity)

# ================== HRG Complexity Mapping ==================
class HRGComplexity:
    """Maps HRG codes to complexity and service times"""
    
    COMPLEXITY_MAP = {
        'BZ33Z': {'name': 'Minor Cataract', 'complexity': 1, 'mean_time': 7.5, 'std_time': 2.5},
        'BZ32B': {'name': 'Intermediate CC 0-1', 'complexity': 2, 'mean_time': 12.5, 'std_time': 2.5},
        'BZ32A': {'name': 'Intermediate CC 2+', 'complexity': 3, 'mean_time': 17.5, 'std_time': 2.5},
        'BZ34B': {'name': 'Phaco CC 0-1', 'complexity': 4, 'mean_time': 10, 'std_time': 2},
        'BZ34A': {'name': 'Phaco CC 2-3', 'complexity': 5, 'mean_time': 12.5, 'std_time': 2.5},
        'BZ35Z': {'name': 'Phaco CC 4+', 'complexity': 6, 'mean_time': 17.5, 'std_time': 2.5},
        'BZ30A': {'name': 'Complex CC 2+', 'complexity': 7, 'mean_time': 25, 'std_time': 5},
        'BZ31B': {'name': 'Very Major CC 0-1', 'complexity': 8, 'mean_time': 35, 'std_time': 5},
        'BZ31A': {'name': 'Very Major CC 2+', 'complexity': 9, 'mean_time': 50, 'std_time': 10},
    }
    
    @classmethod
    def get_service_time(cls, hrg_code: str, random_state=None) -> float:
        """Get service time for HRG code with variation"""
        if random_state is None:
            random_state = np.random
            
        hrg_info = cls.COMPLEXITY_MAP.get(hrg_code, cls.COMPLEXITY_MAP['BZ34B'])
        mean_time = hrg_info['mean_time']
        std_time = hrg_info['std_time']
        
        # Log-normal distribution for service times (always positive, right-skewed)
        service_time = random_state.lognormal(np.log(mean_time), std_time/mean_time)
        return max(5, service_time)  # Minimum 5 minutes

# ================== Simulation Components ==================
class PatientArrivalProcess:
    """Generates patient arrivals based on historical data"""
    
    def __init__(self, env: simpy.Environment, patient_data: pd.DataFrame, 
                 providers: Dict[str, Provider], random_state=None):
        self.env = env
        self.patient_data = patient_data
        self.providers = providers
        self.random_state = random_state or np.random.RandomState(Config.RANDOM_SEED)
        self.patients_generated = []
        
    def generate_arrivals(self, system_type='fragmented'):
        """Generate patient arrivals"""
        # Convert arrival dates to simulation time
        arrivals = []
        for _, row in self.patient_data.iterrows():
            # Calculate arrival time from start_clock_date
            arrival_date = pd.to_datetime(row['Start_clock_date'])
            arrival_day = (arrival_date - pd.to_datetime('2024-01-01')).days % Config.SIM_DURATION
            
            patient = Patient(
                id=str(row['Local Patient Identifier']),
                hrg=row['HRG'],
                hrg_description=row['HRG Description'],
                priority=row['Priority'],
                arrival_time=arrival_day * 24 * 60,  # Convert to minutes
                start_clock_date=arrival_date,
                provider=row['Provider'] if system_type == 'fragmented' else None,
                activity_type=row['Activity Type'],
                lsoa=row['LSOA']
            )
            arrivals.append(patient)
        
        # Sort by arrival time
        arrivals.sort(key=lambda x: x.arrival_time)
        return arrivals

class TreatmentProcess:
    """Handles patient treatment at providers"""
    
    def __init__(self, env: simpy.Environment, provider: Provider, 
                 queue: List[Patient], random_state=None):
        self.env = env
        self.provider = provider
        self.queue = queue
        self.random_state = random_state or np.random.RandomState(Config.RANDOM_SEED)
        
    def treat_patient(self, patient: Patient):
        """Process patient treatment"""
        with self.provider.theatres.request() as req:
            # Wait for theatre availability
            yield req
            
            # Record wait time
            patient.wait_time = self.env.now - patient.arrival_time
            patient.treatment_start = self.env.now
            
            # Get service time based on HRG complexity
            service_time = HRGComplexity.get_service_time(patient.hrg, self.random_state)
            patient.treatment_time = service_time
            
            # Perform treatment
            yield self.env.timeout(service_time)
            
            # Update metrics
            patient.treatment_end = self.env.now
            patient.assigned_provider = self.provider.name
            self.provider.patients_treated += 1
            self.provider.total_wait_time += patient.wait_time
            self.provider.utilization_time += service_time

class SimulationModel:
    """Main simulation model"""
    
    def __init__(self, patient_data: pd.DataFrame, provider_data: pd.DataFrame,
                 system_type='fragmented', random_seed=None):
        self.patient_data = patient_data
        self.provider_data = provider_data
        self.system_type = system_type
        self.random_seed = random_seed or Config.RANDOM_SEED
        self.random_state = np.random.RandomState(self.random_seed)
        
        # Results storage
        self.results = {
            'patients': [],
            'providers': {},
            'kpis': {}
        }
        
    def setup_providers(self, env: simpy.Environment) -> Dict[str, Provider]:
        """Initialize providers with resources"""
        providers = {}
        
        for _, row in self.provider_data.iterrows():
            # Skip providers with no capacity
            if pd.isna(row['annual_volume_2025']) or row['annual_volume_2025'] == 0:
                continue
                
            provider = Provider(
                name=row['provider'],
                provider_type=row['provider_type'],
                annual_capacity=int(row['annual_volume_2025']),
                daily_capacity=row['current_daily_throughput'] or row['annual_volume_2025']/250,
                current_daily_throughput=row['current_daily_throughput'] or 0,
                procedures_per_hour=row['procedures_per_hour_per_theatre'] or 2,
                weighted_service_time=row['weighted_service_time_minutes'] or 15,
                operating_days=row['annual_operating_days'] or 250
            )
            
            # Calculate theatre capacity based on utilization
            if provider.provider_type == 'NHS':
                effective_capacity = provider.daily_capacity / Config.NHS_UTILIZATION
                wait_weeks = Config.NHS_CURRENT_WAIT
            else:
                effective_capacity = provider.daily_capacity / Config.INDEPENDENT_UTILIZATION
                wait_weeks = Config.INDEPENDENT_CURRENT_WAIT
            
            # Set up theatre resources (convert daily to concurrent capacity)
            n_theatres = max(1, int(effective_capacity / (8 * provider.procedures_per_hour)))
            provider.theatres = simpy.Resource(env, capacity=n_theatres)
            
            providers[provider.name] = provider
            
        return providers
    
    def initialize_backlog(self, providers: Dict[str, Provider]) -> List[Patient]:
        """Create initial backlog based on current wait times"""
        backlog = []
        
        for provider in providers.values():
            if provider.provider_type == 'NHS':
                n_backlog = provider.calculate_backlog(Config.NHS_CURRENT_WAIT)
            else:
                n_backlog = provider.calculate_backlog(Config.INDEPENDENT_CURRENT_WAIT)
            
            # Create synthetic backlog patients
            for i in range(n_backlog):
                # Random priority with weights matching data distribution
                priority = self.random_state.choice([1, 2, 3], p=[0.97, 0.025, 0.005])
                
                # Random HRG
                hrg = self.random_state.choice(list(HRGComplexity.COMPLEXITY_MAP.keys()))
                
                # Staggered historical arrival times
                days_ago = self.random_state.uniform(0, 
                    Config.NHS_CURRENT_WAIT * 7 if provider.provider_type == 'NHS' 
                    else Config.INDEPENDENT_CURRENT_WAIT * 7)
                
                patient = Patient(
                    id=f"BACKLOG_{provider.name}_{i}",
                    hrg=hrg,
                    hrg_description=HRGComplexity.COMPLEXITY_MAP[hrg]['name'],
                    priority=priority,
                    arrival_time=-days_ago * 24 * 60,  # Negative time for backlog
                    start_clock_date=datetime.now() - timedelta(days=days_ago),
                    provider=provider.name if self.system_type == 'fragmented' else None
                )
                backlog.append(patient)
        
        return backlog
    
    def run_fragmented_system(self, env: simpy.Environment, providers: Dict[str, Provider],
                            arrivals: List[Patient], backlog: List[Patient]):
        """Run simulation with fragmented PTLs"""
        # Create separate queues for each provider
        provider_queues = {name: [] for name in providers.keys()}
        
        # Assign backlog to providers
        for patient in backlog:
            if patient.provider in provider_queues:
                provider_queues[patient.provider].append(patient)
        
        # Sort queues by priority and arrival time
        for queue in provider_queues.values():
            queue.sort(reverse=True)  # Uses Patient.__lt__
        
        # Process existing backlog and new arrivals
        def patient_flow():
            # Process backlog first
            for provider_name, queue in provider_queues.items():
                provider = providers[provider_name]
                for patient in queue:
                    env.process(TreatmentProcess(env, provider, queue, self.random_state)
                              .treat_patient(patient))
            
            # Process new arrivals
            for patient in arrivals:
                yield env.timeout(max(0, patient.arrival_time - env.now))
                
                # Assign to original provider or closest available
                if patient.provider in providers:
                    assigned_provider = patient.provider
                else:
                    # Find provider with shortest queue
                    min_queue_len = float('inf')
                    assigned_provider = None
                    for name, queue in provider_queues.items():
                        if len(queue) < min_queue_len:
                            min_queue_len = len(queue)
                            assigned_provider = name
                
                if assigned_provider:
                    provider_queues[assigned_provider].append(patient)
                    provider_queues[assigned_provider].sort(reverse=True)
                    env.process(TreatmentProcess(env, providers[assigned_provider], 
                                                provider_queues[assigned_provider], 
                                                self.random_state).treat_patient(patient))
        
        env.process(patient_flow())
    
    def run_consolidated_system(self, env: simpy.Environment, providers: Dict[str, Provider],
                               arrivals: List[Patient], backlog: List[Patient]):
        """Run simulation with consolidated PTL"""
        # Single queue for all patients
        global_queue = []
        
        # Add all backlog to global queue
        global_queue.extend(backlog)
        global_queue.sort(reverse=True)  # Sort by priority and arrival time
        
        # Dynamic allocation process
        def allocate_patients():
            while True:
                if global_queue:
                    # Find provider with shortest expected wait
                    best_provider = None
                    min_expected_wait = float('inf')
                    
                    for provider in providers.values():
                        # Estimate wait based on queue and capacity
                        queue_load = provider.theatres.count + len(provider.theatres.queue)
                        expected_wait = queue_load * provider.weighted_service_time
                        
                        # Prefer providers with lower utilization
                        if provider.provider_type == 'NHS':
                            expected_wait *= Config.NHS_UTILIZATION
                        else:
                            expected_wait *= Config.INDEPENDENT_UTILIZATION
                        
                        if expected_wait < min_expected_wait:
                            min_expected_wait = expected_wait
                            best_provider = provider
                    
                    if best_provider:
                        patient = global_queue.pop(0)
                        env.process(TreatmentProcess(env, best_provider, global_queue, 
                                                    self.random_state).treat_patient(patient))
                
                yield env.timeout(1)  # Check every minute
        
        # Process new arrivals
        def arrival_process():
            for patient in arrivals:
                yield env.timeout(max(0, patient.arrival_time - env.now))
                global_queue.append(patient)
                global_queue.sort(reverse=True)
        
        env.process(allocate_patients())
        env.process(arrival_process())
    
    def run_simulation(self):
        """Main simulation execution"""
        env = simpy.Environment()
        
        # Setup
        providers = self.setup_providers(env)
        arrival_process = PatientArrivalProcess(env, self.patient_data, providers, self.random_state)
        arrivals = arrival_process.generate_arrivals(self.system_type)
        backlog = self.initialize_backlog(providers)
        
        # Run appropriate system
        if self.system_type == 'fragmented':
            self.run_fragmented_system(env, providers, arrivals, backlog)
        else:
            self.run_consolidated_system(env, providers, arrivals, backlog)
        
        # Run simulation
        env.run(until=Config.SIM_DURATION * 24 * 60)  # Convert days to minutes
        
        # Collect results
        self.collect_results(providers, arrivals + backlog)
        
        return self.results
    
    def collect_results(self, providers: Dict[str, Provider], all_patients: List[Patient]):
        """Collect simulation results"""
        # Patient metrics
        treated_patients = [p for p in all_patients if p.treatment_end is not None]
        
        if treated_patients:
            wait_times = [p.wait_time for p in treated_patients]
            self.results['kpis']['avg_wait_time'] = np.mean(wait_times) / (24 * 60)  # Convert to days
            self.results['kpis']['median_wait_time'] = np.median(wait_times) / (24 * 60)
            self.results['kpis']['95th_percentile_wait'] = np.percentile(wait_times, 95) / (24 * 60)
            self.results['kpis']['patients_treated'] = len(treated_patients)
        
        # Provider metrics
        for name, provider in providers.items():
            if provider.patients_treated > 0:
                self.results['providers'][name] = {
                    'patients_treated': provider.patients_treated,
                    'avg_wait_time': provider.total_wait_time / provider.patients_treated / (24 * 60),
                    'utilization': provider.utilization_time / (Config.SIM_DURATION * 24 * 60),
                    'throughput': provider.patients_treated / Config.SIM_DURATION
                }
        
        # Equity metrics (variation in wait times across providers)
        if self.results['providers']:
            provider_waits = [p['avg_wait_time'] for p in self.results['providers'].values()]
            self.results['kpis']['wait_time_std'] = np.std(provider_waits)
            self.results['kpis']['wait_time_cv'] = np.std(provider_waits) / np.mean(provider_waits) if np.mean(provider_waits) > 0 else 0

# ================== Parallel Processing ==================
def run_single_replication(args):
    """Run a single simulation replication"""
    rep_id, patient_data, provider_data, system_type, seed = args
    
    logger.info(f"Starting replication {rep_id} for {system_type} system")
    
    model = SimulationModel(patient_data, provider_data, system_type, seed)
    results = model.run_simulation()
    
    logger.info(f"Completed replication {rep_id} for {system_type} system")
    
    return rep_id, results

def run_parallel_simulation(patient_data: pd.DataFrame, provider_data: pd.DataFrame,
                          system_type='fragmented', n_replications=Config.N_REPLICATIONS):
    """Run multiple replications in parallel"""
    
    # Prepare arguments for parallel execution
    args_list = [
        (i, patient_data, provider_data, system_type, Config.RANDOM_SEED + i)
        for i in range(n_replications)
    ]
    
    results = []
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=Config.N_CORES) as executor:
        futures = [executor.submit(run_single_replication, args) for args in args_list]
        
        for future in as_completed(futures):
            try:
                rep_id, rep_results = future.result()
                results.append(rep_results)
                logger.info(f"Collected results for replication {rep_id}")
            except Exception as e:
                logger.error(f"Error in replication: {e}")
    
    return aggregate_results(results)

def aggregate_results(results_list: List[Dict]) -> Dict:
    """Aggregate results from multiple replications"""
    aggregated = {
        'kpis': {},
        'providers': {},
        'confidence_intervals': {}
    }
    
    # Aggregate KPIs
    kpi_names = results_list[0]['kpis'].keys() if results_list else []
    for kpi in kpi_names:
        values = [r['kpis'][kpi] for r in results_list if kpi in r['kpis']]
        if values:
            aggregated['kpis'][kpi] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
            
            # 95% confidence interval
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            aggregated['confidence_intervals'][kpi] = (ci_lower, ci_upper)
    
    # Aggregate provider metrics
    all_providers = set()
    for r in results_list:
        all_providers.update(r.get('providers', {}).keys())
    
    for provider in all_providers:
        provider_metrics = []
        for r in results_list:
            if provider in r.get('providers', {}):
                provider_metrics.append(r['providers'][provider])
        
        if provider_metrics:
            aggregated['providers'][provider] = {}
            for metric in provider_metrics[0].keys():
                values = [p[metric] for p in provider_metrics]
                aggregated['providers'][provider][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
    
    return aggregated

# ================== Scenario Testing ==================
class ScenarioAnalyzer:
    """Analyze different operational scenarios"""
    
    def __init__(self, patient_data: pd.DataFrame, provider_data: pd.DataFrame):
        self.patient_data = patient_data
        self.provider_data = provider_data
        self.scenarios = {}
        
    def run_baseline_comparison(self):
        """Compare fragmented vs consolidated systems"""
        logger.info("Running baseline comparison: Fragmented vs Consolidated")
        
        # Run fragmented system
        logger.info("Simulating fragmented system...")
        fragmented_results = run_parallel_simulation(
            self.patient_data, self.provider_data, 'fragmented'
        )
        
        # Run consolidated system
        logger.info("Simulating consolidated system...")
        consolidated_results = run_parallel_simulation(
            self.patient_data, self.provider_data, 'consolidated'
        )
        
        return {
            'fragmented': fragmented_results,
            'consolidated': consolidated_results,
            'improvement': self.calculate_improvements(fragmented_results, consolidated_results)
        }
    
    def run_capacity_scenarios(self):
        """Test different capacity utilization scenarios"""
        scenarios = {
            'current': (Config.NHS_UTILIZATION, Config.INDEPENDENT_UTILIZATION),
            'optimized_nhs': (0.95, Config.INDEPENDENT_UTILIZATION),
            'optimized_independent': (Config.NHS_UTILIZATION, 0.90),
            'fully_optimized': (0.95, 0.90)
        }
        
        results = {}
        for scenario_name, (nhs_util, ind_util) in scenarios.items():
            logger.info(f"Running scenario: {scenario_name}")
            
            # Temporarily update utilization
            original_nhs = Config.NHS_UTILIZATION
            original_ind = Config.INDEPENDENT_UTILIZATION
            Config.NHS_UTILIZATION = nhs_util
            Config.INDEPENDENT_UTILIZATION = ind_util
            
            results[scenario_name] = run_parallel_simulation(
                self.patient_data, self.provider_data, 'consolidated'
            )
            
            # Restore original values
            Config.NHS_UTILIZATION = original_nhs
            Config.INDEPENDENT_UTILIZATION = original_ind
        
        return results
    
    def run_weekend_sessions(self):
        """Test impact of weekend sessions"""
        # Modify provider data to add weekend capacity
        modified_provider_data = self.provider_data.copy()
        modified_provider_data['annual_operating_days'] = modified_provider_data['annual_operating_days'] * 1.2
        modified_provider_data['annual_volume_2025'] = modified_provider_data['annual_volume_2025'] * 1.2
        
        logger.info("Running weekend sessions scenario...")
        return run_parallel_simulation(
            self.patient_data, modified_provider_data, 'consolidated'
        )
    
    def calculate_improvements(self, baseline: Dict, improved: Dict) -> Dict:
        """Calculate percentage improvements"""
        improvements = {}
        
        for kpi in baseline['kpis']:
            if kpi in improved['kpis']:
                baseline_val = baseline['kpis'][kpi]['mean']
                improved_val = improved['kpis'][kpi]['mean']
                
                if baseline_val != 0:
                    pct_change = ((improved_val - baseline_val) / baseline_val) * 100
                    improvements[kpi] = {
                        'baseline': baseline_val,
                        'improved': improved_val,
                        'change_pct': pct_change,
                        'change_abs': improved_val - baseline_val
                    }
        
        return improvements

# ================== Visualization and Reporting ==================
class SimulationReporter:
    """Generate comprehensive reports from simulation results"""
    
    @staticmethod
    def generate_summary_report(results: Dict, output_file='simulation_report.txt'):
        """Generate text summary report"""
        report_lines = [
            "=" * 80,
            "PATIENT TREATMENT LIST CONSOLIDATION SIMULATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Simulation Duration: {Config.SIM_DURATION} days",
            f"Number of Replications: {Config.N_REPLICATIONS}",
            f"Parallel Cores Used: {Config.N_CORES}",
            "",
            "=" * 80,
            "KEY PERFORMANCE INDICATORS",
            "=" * 80,
        ]
        
        # Add KPI results
        for kpi_name, kpi_data in results.get('kpis', {}).items():
            if isinstance(kpi_data, dict):
                report_lines.append(f"\n{kpi_name.replace('_', ' ').title()}:")
                report_lines.append(f"  Mean: {kpi_data['mean']:.2f}")
                report_lines.append(f"  Std Dev: {kpi_data['std']:.2f}")
                report_lines.append(f"  Range: [{kpi_data['min']:.2f}, {kpi_data['max']:.2f}]")
                
                if kpi_name in results.get('confidence_intervals', {}):
                    ci = results['confidence_intervals'][kpi_name]
                    report_lines.append(f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
        
        # Add provider-specific results
        if 'providers' in results:
            report_lines.extend([
                "",
                "=" * 80,
                "PROVIDER-SPECIFIC METRICS",
                "=" * 80,
            ])
            
            for provider, metrics in results['providers'].items():
                report_lines.append(f"\n{provider}:")
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict):
                        report_lines.append(f"  {metric_name}: {metric_data['mean']:.2f} (±{metric_data['std']:.2f})")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return '\n'.join(report_lines)
    
    @staticmethod
    def generate_comparison_report(comparison_results: Dict, output_file='comparison_report.txt'):
        """Generate comparison report between scenarios"""
        report_lines = [
            "=" * 80,
            "SCENARIO COMPARISON REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        if 'improvement' in comparison_results:
            report_lines.extend([
                "=" * 80,
                "FRAGMENTED vs CONSOLIDATED SYSTEM IMPROVEMENTS",
                "=" * 80,
            ])
            
            for metric, improvement in comparison_results['improvement'].items():
                report_lines.append(f"\n{metric.replace('_', ' ').title()}:")
                report_lines.append(f"  Baseline (Fragmented): {improvement['baseline']:.2f}")
                report_lines.append(f"  Improved (Consolidated): {improvement['improved']:.2f}")
                report_lines.append(f"  Change: {improvement['change_abs']:.2f} ({improvement['change_pct']:+.1f}%)")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return '\n'.join(report_lines)
    
    @staticmethod
    def export_results_to_csv(results: Dict, prefix='simulation'):
        """Export results to CSV files for further analysis"""
        # Export KPIs
        if 'kpis' in results:
            kpi_df = pd.DataFrame(results['kpis']).T
            kpi_df.to_csv(f'{prefix}_kpis.csv')
        
        # Export provider metrics
        if 'providers' in results:
            provider_data = []
            for provider, metrics in results['providers'].items():
                row = {'provider': provider}
                for metric, values in metrics.items():
                    if isinstance(values, dict):
                        row[f'{metric}_mean'] = values['mean']
                        row[f'{metric}_std'] = values['std']
                    else:
                        row[metric] = values
                provider_data.append(row)
            
            provider_df = pd.DataFrame(provider_data)
            provider_df.to_csv(f'{prefix}_providers.csv', index=False)

# ================== Main Execution ==================
def main():
    """Main execution function"""
    start_time = time.time()
    
    logger.info(f"Starting PTL Consolidation Simulation on {Config.N_CORES} cores")
    
    # Load data
    logger.info("Loading data...")
    patient_data = pd.read_csv('new_data_ref_dates.csv')
    provider_data = pd.read_csv('provider_capacity_data.csv')
    
    logger.info(f"Loaded {len(patient_data)} patient records and {len(provider_data)} providers")
    
    # Initialize analyzer
    analyzer = ScenarioAnalyzer(patient_data, provider_data)
    
    # Run baseline comparison
    logger.info("Running baseline comparison...")
    baseline_results = analyzer.run_baseline_comparison()
    
    # Generate reports
    reporter = SimulationReporter()
    
    # Generate fragmented system report
    fragmented_report = reporter.generate_summary_report(
        baseline_results['fragmented'], 
        'fragmented_system_report.txt'
    )
    logger.info("Fragmented system report generated")
    
    # Generate consolidated system report
    consolidated_report = reporter.generate_summary_report(
        baseline_results['consolidated'], 
        'consolidated_system_report.txt'
    )
    logger.info("Consolidated system report generated")
    
    # Generate comparison report
    comparison_report = reporter.generate_comparison_report(
        baseline_results, 
        'system_comparison_report.txt'
    )
    logger.info("Comparison report generated")
    
    # Export detailed results to CSV
    reporter.export_results_to_csv(baseline_results['fragmented'], 'fragmented')
    reporter.export_results_to_csv(baseline_results['consolidated'], 'consolidated')
    
    # Run additional scenarios
    logger.info("Running capacity optimization scenarios...")
    capacity_results = analyzer.run_capacity_scenarios()
    
    # Generate capacity scenario reports
    for scenario_name, results in capacity_results.items():
        reporter.generate_summary_report(
            results, 
            f'capacity_scenario_{scenario_name}_report.txt'
        )
        reporter.export_results_to_csv(results, f'capacity_{scenario_name}')
    
    # Run weekend sessions scenario
    logger.info("Running weekend sessions scenario...")
    weekend_results = analyzer.run_weekend_sessions()
    reporter.generate_summary_report(weekend_results, 'weekend_sessions_report.txt')
    reporter.export_results_to_csv(weekend_results, 'weekend_sessions')
    
    # Calculate total execution time
    execution_time = time.time() - start_time
    logger.info(f"Simulation completed in {execution_time:.2f} seconds")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Reports Generated: 7")
    print(f"CSV Files Created: 10")
    print("\nKey Findings:")
    
    if 'improvement' in baseline_results:
        for metric, improvement in baseline_results['improvement'].items():
            if 'wait' in metric.lower():
                print(f"  {metric.replace('_', ' ').title()}: {improvement['change_pct']:+.1f}%")
    
    print("\nFiles created:")
    print("  - fragmented_system_report.txt")
    print("  - consolidated_system_report.txt")
    print("  - system_comparison_report.txt")
    print("  - capacity_scenario_*_report.txt")
    print("  - weekend_sessions_report.txt")
    print("  - *.csv (detailed results)")

if __name__ == "__main__":
    # Set up multiprocessing for Windows compatibility
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    main()
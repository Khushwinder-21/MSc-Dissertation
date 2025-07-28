import numpy as np
import pandas as pd
import simpy
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class HealthcareQueueingModel:
    """
    Mathematical model for healthcare queueing system using M/M/c queue theory
    """
    
    def __init__(self, arrival_rate: float, service_rate: float, num_servers: int):
        """
        Initialize M/M/c queue parameters
        
        Args:
            arrival_rate (λ): Average patient arrivals per time unit
            service_rate (μ): Average service completions per server per time unit
            num_servers (c): Number of servers (surgeons/theatres)
        """
        self.lambda_ = arrival_rate
        self.mu = service_rate
        self.c = num_servers
        self.rho = arrival_rate / (num_servers * service_rate)  # Traffic intensity
        
    def calculate_p0(self) -> float:
        """Calculate probability of zero patients in system"""
        sum_term = sum([(self.lambda_/self.mu)**n / np.math.factorial(n) 
                       for n in range(self.c)])
        
        if self.rho < 1:  # System is stable
            last_term = ((self.lambda_/self.mu)**self.c / np.math.factorial(self.c)) * (1/(1-self.rho))
            p0 = 1 / (sum_term + last_term)
        else:
            p0 = 0  # System is unstable
            
        return p0
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate key performance metrics using Little's Law"""
        if self.rho >= 1:
            return {
                'utilization': self.rho,
                'avg_queue_length': float('inf'),
                'avg_system_length': float('inf'),
                'avg_waiting_time': float('inf'),
                'avg_system_time': float('inf'),
                'probability_wait': 1.0
            }
        
        p0 = self.calculate_p0()
        
        # Erlang C formula - probability of waiting
        pc = ((self.lambda_/self.mu)**self.c / np.math.factorial(self.c)) * p0 / (1 - self.rho)
        
        # Average queue length (Lq)
        lq = pc * self.rho / (1 - self.rho)
        
        # Average number in system (L)
        l = lq + self.lambda_/self.mu
        
        # Average waiting time in queue (Wq) - Little's Law
        wq = lq / self.lambda_
        
        # Average time in system (W)
        w = l / self.lambda_
        
        return {
            'utilization': self.rho,
            'avg_queue_length': lq,
            'avg_system_length': l,
            'avg_waiting_time': wq,
            'avg_system_time': w,
            'probability_wait': pc
        }

class Patient:
    """Patient entity for simulation"""
    
    def __init__(self, patient_id: int, arrival_time: float, provider_id: str, 
                 procedure_type: str, priority: int = 3):
        self.id = patient_id
        self.arrival_time = arrival_time
        self.provider_id = provider_id
        self.procedure_type = procedure_type
        self.priority = priority
        self.start_service_time = None
        self.end_service_time = None
        self.waiting_time = None
        self.service_time = None
        
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority < other.priority

class Provider:
    """Healthcare provider with resources"""
    
    def __init__(self, provider_id: str, num_theatres: int, 
                 service_rates: Dict[str, float], capacity_utilization: float = 0.85):
        self.provider_id = provider_id
        self.num_theatres = num_theatres
        self.service_rates = service_rates  # Service rate per procedure type
        self.capacity_utilization = capacity_utilization
        self.patients_served = 0
        self.total_waiting_time = 0
        self.total_service_time = 0
        
class DiscreteEventSimulation:
    """
    Discrete Event Simulation for healthcare system
    """
    
    def __init__(self, providers: List[Provider], simulation_time: int = 365*24):
        """
        Initialize simulation
        
        Args:
            providers: List of Provider objects
            simulation_time: Simulation duration in hours (default: 1 year)
        """
        self.env = simpy.Environment()
        self.providers = {p.provider_id: p for p in providers}
        self.simulation_time = simulation_time
        self.patient_counter = 0
        self.results = defaultdict(list)
        self.consolidated_mode = False
        
    def generate_patient_arrivals(self, arrival_rates: Dict[str, Dict[str, float]]):
        """
        Generate patient arrivals based on Poisson process
        
        Args:
            arrival_rates: Dict of {provider_id: {procedure_type: rate}}
        """
        while True:
            for provider_id, procedures in arrival_rates.items():
                for procedure_type, rate in procedures.items():
                    # Poisson arrivals
                    if rate > 0:
                        inter_arrival_time = np.random.exponential(1/rate)
                        yield self.env.timeout(inter_arrival_time)
                        
                        patient = Patient(
                            patient_id=self.patient_counter,
                            arrival_time=self.env.now,
                            provider_id=provider_id,
                            procedure_type=procedure_type,
                            priority=np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.6, 0.15, 0.05])
                        )
                        self.patient_counter += 1
                        
                        if self.consolidated_mode:
                            # Route to best provider
                            best_provider = self.find_best_provider(procedure_type)
                            patient.provider_id = best_provider
                            
                        self.env.process(self.patient_flow(patient))
    
    def find_best_provider(self, procedure_type: str) -> str:
        """Find provider with shortest expected wait time"""
        min_wait = float('inf')
        best_provider = None
        
        for provider_id, provider in self.providers.items():
            if procedure_type in provider.service_rates:
                # Estimate current queue length
                queue_length = len([p for p in self.results['waiting_patients'] 
                                  if p.provider_id == provider_id])
                expected_wait = queue_length / (provider.num_theatres * provider.service_rates[procedure_type])
                
                if expected_wait < min_wait:
                    min_wait = expected_wait
                    best_provider = provider_id
                    
        return best_provider or list(self.providers.keys())[0]
    
    def patient_flow(self, patient: Patient):
        """Simulate patient flow through the system"""
        provider = self.providers[patient.provider_id]
        
        # Request theatre resource
        with provider.theatre_resource.request() as request:
            # Record patient entering queue
            self.results['waiting_patients'].append(patient)
            
            # Wait for theatre
            yield request
            
            # Remove from waiting list
            self.results['waiting_patients'].remove(patient)
            
            # Record service start
            patient.start_service_time = self.env.now
            patient.waiting_time = patient.start_service_time - patient.arrival_time
            
            # Service time (exponential distribution)
            service_rate = provider.service_rates.get(patient.procedure_type, 0.5)
            service_time = np.random.exponential(1/service_rate)
            patient.service_time = service_time
            
            yield self.env.timeout(service_time)
            
            # Record service completion
            patient.end_service_time = self.env.now
            
            # Update provider statistics
            provider.patients_served += 1
            provider.total_waiting_time += patient.waiting_time
            provider.total_service_time += patient.service_time
            
            # Store results
            self.results['completed_patients'].append(patient)
    
    def run_simulation(self, arrival_rates: Dict[str, Dict[str, float]], 
                      consolidated: bool = False) -> pd.DataFrame:
        """
        Run the simulation
        
        Args:
            arrival_rates: Patient arrival rates by provider and procedure
            consolidated: Whether to use consolidated PTL mode
        """
        self.consolidated_mode = consolidated
        self.results = defaultdict(list)
        self.env = simpy.Environment()
        
        # Create theatre resources for each provider
        for provider in self.providers.values():
            provider.theatre_resource = simpy.Resource(self.env, capacity=provider.num_theatres)
            provider.patients_served = 0
            provider.total_waiting_time = 0
            provider.total_service_time = 0
        
        # Start patient generation process
        self.env.process(self.generate_patient_arrivals(arrival_rates))
        
        # Run simulation
        self.env.run(until=self.simulation_time)
        
        # Convert results to DataFrame
        results_data = []
        for patient in self.results['completed_patients']:
            results_data.append({
                'patient_id': patient.id,
                'provider_id': patient.provider_id,
                'procedure_type': patient.procedure_type,
                'priority': patient.priority,
                'arrival_time': patient.arrival_time,
                'start_service_time': patient.start_service_time,
                'end_service_time': patient.end_service_time,
                'waiting_time': patient.waiting_time,
                'service_time': patient.service_time,
                'total_time': patient.waiting_time + patient.service_time
            })
            
        return pd.DataFrame(results_data)

class PTLConsolidationAnalyzer:
    """
    Analyzer for comparing fragmented vs consolidated PTL systems
    """
    
    def __init__(self, provider_data: pd.DataFrame):
        """
        Initialize analyzer with provider data
        
        Args:
            provider_data: DataFrame with provider information
        """
        self.provider_data = provider_data
        self.setup_providers()
        
    def setup_providers(self):
        """Setup provider objects based on data"""
        self.providers = []
        
        # Group by provider to get unique providers
        provider_summary = self.provider_data.groupby('Provider').agg({
            'Activity': 'count',
            'Procedure': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        for _, row in provider_summary.iterrows():
            # Estimate number of theatres based on activity volume
            annual_procedures = row['Activity']
            estimated_theatres = max(1, int(annual_procedures / (250 * 4)))  # 250 days, 4 procedures/day/theatre
            
            # Calculate service rates for each procedure type
            procedure_counts = row['Procedure']
            total_procedures = sum(procedure_counts.values())
            service_rates = {}
            
            for procedure, count in procedure_counts.items():
                # Estimate service rate (procedures per hour)
                # Assuming average procedure time of 1-3 hours
                avg_procedure_time = np.random.uniform(1, 3)
                service_rates[procedure] = 1 / avg_procedure_time
            
            provider = Provider(
                provider_id=row['Provider'],
                num_theatres=estimated_theatres,
                service_rates=service_rates
            )
            self.providers.append(provider)
    
    def calculate_arrival_rates(self) -> Dict[str, Dict[str, float]]:
        """Calculate arrival rates from historical data"""
        arrival_rates = {}
        
        # Calculate monthly procedure counts
        monthly_counts = self.provider_data.groupby(['Provider', 'Procedure']).size()
        
        for (provider, procedure), count in monthly_counts.items():
            if provider not in arrival_rates:
                arrival_rates[provider] = {}
            
            # Convert to hourly rate (assuming 30 days, 24 hours)
            hourly_rate = count / (30 * 24)
            arrival_rates[provider][procedure] = hourly_rate
            
        return arrival_rates
    
    def run_comparison(self, simulation_hours: int = 30*24) -> Dict[str, pd.DataFrame]:
        """
        Run comparison between fragmented and consolidated systems
        
        Args:
            simulation_hours: Duration of simulation in hours
        """
        arrival_rates = self.calculate_arrival_rates()
        
        # Run fragmented system simulation
        print("Running fragmented system simulation...")
        sim_fragmented = DiscreteEventSimulation(self.providers, simulation_hours)
        results_fragmented = sim_fragmented.run_simulation(arrival_rates, consolidated=False)
        
        # Run consolidated system simulation
        print("Running consolidated system simulation...")
        sim_consolidated = DiscreteEventSimulation(self.providers, simulation_hours)
        results_consolidated = sim_consolidated.run_simulation(arrival_rates, consolidated=True)
        
        return {
            'fragmented': results_fragmented,
            'consolidated': results_consolidated
        }
    
    def analyze_results(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Analyze and compare simulation results"""
        comparison = []
        
        for system_type, df in results.items():
            if len(df) > 0:
                metrics = {
                    'system_type': system_type,
                    'total_patients': len(df),
                    'avg_waiting_time': df['waiting_time'].mean(),
                    'median_waiting_time': df['waiting_time'].median(),
                    'max_waiting_time': df['waiting_time'].max(),
                    'avg_service_time': df['service_time'].mean(),
                    'avg_total_time': df['total_time'].mean(),
                    'std_waiting_time': df['waiting_time'].std()
                }
                
                # Calculate percentiles
                for p in [75, 90, 95]:
                    metrics[f'p{p}_waiting_time'] = df['waiting_time'].quantile(p/100)
                
                # Provider-level metrics
                provider_metrics = df.groupby('provider_id').agg({
                    'waiting_time': ['mean', 'count'],
                    'service_time': 'mean'
                })
                
                metrics['provider_variance'] = provider_metrics['waiting_time']['mean'].var()
                
                comparison.append(metrics)
        
        return pd.DataFrame(comparison)
    
    def visualize_results(self, results: Dict[str, pd.DataFrame]):
        """Create visualizations comparing systems"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Waiting time distribution
        ax = axes[0, 0]
        for system_type, df in results.items():
            if len(df) > 0:
                ax.hist(df['waiting_time'], bins=50, alpha=0.6, label=system_type, density=True)
        ax.set_xlabel('Waiting Time (hours)')
        ax.set_ylabel('Density')
        ax.set_title('Waiting Time Distribution')
        ax.legend()
        
        # 2. Waiting time by provider
        ax = axes[0, 1]
        for system_type, df in results.items():
            if len(df) > 0:
                provider_waits = df.groupby('provider_id')['waiting_time'].mean().sort_values()
                ax.bar(range(len(provider_waits)), provider_waits.values, 
                      alpha=0.6, label=system_type)
        ax.set_xlabel('Provider')
        ax.set_ylabel('Average Waiting Time (hours)')
        ax.set_title('Average Waiting Time by Provider')
        ax.legend()
        
        # 3. Cumulative waiting time
        ax = axes[0, 2]
        for system_type, df in results.items():
            if len(df) > 0:
                sorted_waits = np.sort(df['waiting_time'])
                ax.plot(sorted_waits, np.linspace(0, 100, len(sorted_waits)), 
                       label=system_type, linewidth=2)
        ax.set_xlabel('Waiting Time (hours)')
        ax.set_ylabel('Cumulative Percentage')
        ax.set_title('Cumulative Waiting Time Distribution')
        ax.legend()
        
        # 4. Resource utilization (approximate)
        ax = axes[1, 0]
        for system_type, df in results.items():
            if len(df) > 0:
                utilization = df.groupby('provider_id').apply(
                    lambda x: x['service_time'].sum() / (len(x) * x['total_time'].mean())
                )
                ax.bar(range(len(utilization)), utilization.values, 
                      alpha=0.6, label=system_type)
        ax.set_xlabel('Provider')
        ax.set_ylabel('Utilization')
        ax.set_title('Resource Utilization by Provider')
        ax.legend()
        
        # 5. Priority-based waiting times
        ax = axes[1, 1]
        for system_type, df in results.items():
            if len(df) > 0:
                priority_waits = df.groupby('priority')['waiting_time'].mean().sort_index()
                ax.plot(priority_waits.index, priority_waits.values, 
                       marker='o', label=system_type, linewidth=2)
        ax.set_xlabel('Priority (1=Highest)')
        ax.set_ylabel('Average Waiting Time (hours)')
        ax.set_title('Waiting Time by Priority')
        ax.legend()
        
        # 6. Time series of queue length (approximate from completed patients)
        ax = axes[1, 2]
        for system_type, df in results.items():
            if len(df) > 0:
                # Bin arrivals by hour
                df['arrival_hour'] = df['arrival_time'].astype(int)
                queue_length = df.groupby('arrival_hour').size()
                ax.plot(queue_length.index, queue_length.values, 
                       label=system_type, alpha=0.7)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Arrivals per Hour')
        ax.set_title('Patient Arrivals Over Time')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

# Mathematical modeling functions
def create_queueing_models(provider_data: pd.DataFrame) -> Dict[str, HealthcareQueueingModel]:
    """Create M/M/c models for each provider"""
    models = {}
    
    # Calculate parameters for each provider
    provider_stats = provider_data.groupby('Provider').agg({
        'Activity': 'count'
    }).reset_index()
    
    for _, row in provider_stats.iterrows():
        provider = row['Provider']
        annual_procedures = row['Activity']
        
        # Estimate parameters
        daily_arrivals = annual_procedures / 365
        hourly_arrivals = daily_arrivals / 8  # Assuming 8-hour working days
        
        # Estimate number of servers (theatres)
        estimated_theatres = max(1, int(annual_procedures / (250 * 4)))
        
        # Estimate service rate (procedures per hour per theatre)
        service_rate = 0.4  # Approximately 2.5 hours per procedure
        
        model = HealthcareQueueingModel(
            arrival_rate=hourly_arrivals,
            service_rate=service_rate,
            num_servers=estimated_theatres
        )
        
        models[provider] = model
    
    return models

def compare_consolidated_model(models: Dict[str, HealthcareQueueingModel]) -> pd.DataFrame:
    """Compare individual models vs consolidated model"""
    comparison_results = []
    
    # Individual provider metrics
    total_arrival_rate = 0
    total_servers = 0
    
    for provider, model in models.items():
        metrics = model.calculate_metrics()
        metrics['provider'] = provider
        metrics['system'] = 'Fragmented'
        comparison_results.append(metrics)
        
        total_arrival_rate += model.lambda_
        total_servers += model.c
    
    # Consolidated system model
    # Assume same average service rate
    avg_service_rate = np.mean([m.mu for m in models.values()])
    
    consolidated_model = HealthcareQueueingModel(
        arrival_rate=total_arrival_rate,
        service_rate=avg_service_rate,
        num_servers=total_servers
    )
    
    consolidated_metrics = consolidated_model.calculate_metrics()
    consolidated_metrics['provider'] = 'Consolidated'
    consolidated_metrics['system'] = 'Consolidated'
    comparison_results.append(consolidated_metrics)
    
    return pd.DataFrame(comparison_results)

# Placeholder for waiting time calculation when referral dates are available
def calculate_actual_waiting_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate actual waiting times when referral dates become available
    
    Args:
        df: DataFrame with 'referral_date' and 'activity_date' columns
    
    Returns:
        DataFrame with additional 'waiting_time_days' column
    """
    # This function will be used when referral dates are available
    # df['waiting_time_days'] = (df['activity_date'] - df['referral_date']).dt.days
    
    # For now, create synthetic waiting times based on procedure complexity
    procedure_complexity = {
        'Cataract Surgery': 30,  # Average 30 days wait
        'YAG Laser Capsulotomy': 14,
        'Glaucoma Surgery': 45,
        'Intravitreal Injection Therapies': 21,
        'Diagnostic': 7
    }
    
    # Add placeholder column
    df['estimated_waiting_time_days'] = df['Procedure'].map(
        lambda x: procedure_complexity.get(x, 28)  # Default 28 days
    )
    
    # Add some random variation
    df['estimated_waiting_time_days'] += np.random.normal(0, 7, len(df))
    df['estimated_waiting_time_days'] = df['estimated_waiting_time_days'].clip(lower=1)
    
    return df

# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data for demonstration
    print("Creating sample healthcare data...")
    
    # Generate sample provider data
    providers = ['Provider_A', 'Provider_B', 'Provider_C']
    procedures = ['Cataract Surgery', 'YAG Laser Capsulotomy', 'Glaucoma Surgery']
    
    sample_data = []
    for _ in range(1000):
        sample_data.append({
            'Provider': np.random.choice(providers),
            'Procedure': np.random.choice(procedures),
            'Activity': 1,
            'Activity_Date': datetime.now() - timedelta(days=np.random.randint(0, 30))
        })
    
    sample_df = pd.DataFrame(sample_data)
    
    # 1. Mathematical Modeling
    print("\n1. MATHEMATICAL QUEUEING MODELS")
    print("="*50)
    
    models = create_queueing_models(sample_df)
    
    for provider, model in models.items():
        metrics = model.calculate_metrics()
        print(f"\n{provider} Metrics:")
        print(f"  Utilization: {metrics['utilization']:.2%}")
        print(f"  Avg Queue Length: {metrics['avg_queue_length']:.2f}")
        print(f"  Avg Waiting Time: {metrics['avg_waiting_time']:.2f} hours")
        print(f"  Probability of Waiting: {metrics['probability_wait']:.2%}")
    
    # Compare with consolidated model
    print("\n2. CONSOLIDATED VS FRAGMENTED COMPARISON")
    print("="*50)
    
    comparison_df = compare_consolidated_model(models)
    print(comparison_df[['provider', 'system', 'utilization', 
                        'avg_waiting_time', 'avg_queue_length']].round(3))
    
    # 2. Discrete Event Simulation
    print("\n3. DISCRETE EVENT SIMULATION")
    print("="*50)
    
    analyzer = PTLConsolidationAnalyzer(sample_df)
    
    # Run shorter simulation for demonstration
    simulation_results = analyzer.run_comparison(simulation_hours=7*24)  # 1 week
    
    # Analyze results
    analysis_df = analyzer.analyze_results(simulation_results)
    print("\nSimulation Results Comparison:")
    print(analysis_df[['system_type', 'avg_waiting_time', 
                      'median_waiting_time', 'provider_variance']].round(2))
    
    # Calculate improvement metrics
    if len(analysis_df) == 2:
        fragmented = analysis_df[analysis_df['system_type'] == 'fragmented'].iloc[0]
        consolidated = analysis_df[analysis_df['system_type'] == 'consolidated'].iloc[0]
        
        improvement = {
            'waiting_time_reduction': (fragmented['avg_waiting_time'] - consolidated['avg_waiting_time']) / fragmented['avg_waiting_time'] * 100,
            'variance_reduction': (fragmented['provider_variance'] - consolidated['provider_variance']) / fragmented['provider_variance'] * 100,
        }
        
        print(f"\nIMPROVEMENT METRICS:")
        print(f"  Waiting Time Reduction: {improvement['waiting_time_reduction']:.1f}%")
        print(f"  Provider Variance Reduction: {improvement['variance_reduction']:.1f}%")
    
    # Visualize results
    print("\nGenerating visualizations...")
    analyzer.visualize_results(simulation_results)
    
    # 3. Waiting time calculation placeholder
    print("\n4. WAITING TIME CALCULATION (Placeholder)")
    print("="*50)
    
    sample_with_waiting = calculate_actual_waiting_times(sample_df)
    print(f"Average Estimated Waiting Time: {sample_with_waiting['estimated_waiting_time_days'].mean():.1f} days")
    print(f"Median Estimated Waiting Time: {sample_with_waiting['estimated_waiting_time_days'].median():.1f} days")
    
    print("\n✅ Simulation framework ready for actual data integration!")
    print("When referral dates are available, update the calculate_actual_waiting_times function.")
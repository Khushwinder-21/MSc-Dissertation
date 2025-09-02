# M/G/c Queue Analysis
# impoting necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
from scipy.special import factorial
import time
import warnings
warnings.filterwarnings('ignore')
#defining M/G/c queue analysis class
class MGcQueueClass:
    
    def __init__(self):
        self.results = {}
        
       # Assumed utilisation rates for more realistic wait times
        self.assumed_utilisation = {
            'NHS': 0.90,           # 90% for NHS providers
            'Independent': 0.85    # 85% for independent providers
        }
        
        #assumed backlog months for more realistic wait times same across all methods
        self.assumed_backlog_months = {
            'NHS': 9.24,              # 40 weeks
            'Independent': 6.01       # 26 weeks
        }
        
        # Service rate multiplier to reflect more realistic wait times
        self.service_rate_multiplier = 1.3
        
        # Priority distribution
        self.priority_dist = {
            1: 0.7,   # 70% Routine
            2: 0.1,   # 10% Urgent
            3: 0.2    # 20% Two Week Wait
        }
    # calculating erlang c formula for M/M/c queue
    def erlang_c_formula(self, lambda_rate, mu_rate, c):
        if mu_rate == 0 or c == 0 or lambda_rate == 0:
            return {
                'utilization': 0,
                'P_wait': 0,
                'L_q': 0,
                'W_q': 0,
                'stable': True
            }
            
        rho = lambda_rate / (c * mu_rate)
        
        if rho >= 1:
            return {
                'utilization': rho,
                'P_wait': 1,
                'L_q': float('inf'),
                'W_q': float('inf'),
                'stable': False
            }
        
        # Calculating P0 which is the probability that there are 0 patients in the system
        try:
            sum_term = sum([(lambda_rate/mu_rate)**n / factorial(n) for n in range(int(c))])
            sum_term += (lambda_rate/mu_rate)**c / factorial(int(c)) * (1 / (1 - rho))
            P0 = 1 / sum_term
            
            # Erlang C probability
            P_wait = ((lambda_rate/mu_rate)**c / factorial(int(c))) * (1 / (1 - rho)) * P0
            
            # Queue metrics
            L_q = P_wait * rho / (1 - rho)
            W_q = L_q / lambda_rate
        # returns dict with utilization, P_wait, L_q, W_q, stable
        except:
            return {
                'utilization': rho,
                'P_wait': 1,
                'L_q': float('inf'),
                'W_q': float('inf'),
                'stable': False
            }
        
        return {
            'utilization': rho,
            'P_wait': P_wait,
            'L_q': L_q,
            'W_q': W_q,
            'stable': True
        }
    # M/G/c correction using CV of service times
    def mgc_correction(self, mmc_results, cv):
        correction_factor = (1 + cv**2) / 2
        mgc_results = mmc_results.copy()
        mgc_results['W_q'] = mmc_results['W_q'] * correction_factor
        mgc_results['L_q'] = mmc_results['L_q'] * correction_factor
        mgc_results['cv'] = cv
        return mgc_results
    
    def run_analysis(self, provider_data, provider_type):
        """Analysing single provider with adjusted service rates"""
        # Get parameters
        annual_volume = provider_data.get('annual_volume_2024', 0)
        if annual_volume == 0:
            annual_volume = provider_data.get('annual_volume_2023', 0)
        
        if annual_volume == 0:
            return None
        
        # Service parameters with multiplier for more realistic wait times
        daily_throughput = provider_data.get('current_daily_throughput', 0) * self.service_rate_multiplier
        operating_days = provider_data.get('days_per_week', 5)
        theatres = provider_data.get('theatres', 1)
        
        if daily_throughput == 0 or theatres == 0:
            return None
        
        # Calculating rates
        weekly_capacity = daily_throughput * operating_days
        mu_per_server = weekly_capacity / theatres  # Service rate per theatre
        
        # Adjustiing arrival rate for target utilisation
        target_util = self.assumed_utilisation[provider_type]
        lambda_rate = target_util * theatres * mu_per_server
        
        # Coefficient of Variation (CV) assumptions
        cv = 0.8 if provider_type == 'NHS' else 0.6
        
        mmc_results = self.erlang_c_formula(lambda_rate, mu_per_server, theatres) # M/M/c results
        mgc_results = self.mgc_correction(mmc_results, cv) # Apply M/G/c correction
        
        # Calculate reduced backlog
        backlog_weeks = self.assumed_backlog_months[provider_type] * 4.33 # convert months to weeks
        backlog_size = int(lambda_rate * backlog_weeks)
        
        # Backlog clearance time
        if weekly_capacity > lambda_rate:
            efficiency_factor = 0.3  # derived from DES efficiency
            backlog_clear_time = min(efficiency_factor * (backlog_size / max(weekly_capacity - lambda_rate, 1e-3)),52)

        else:
            backlog_clear_time = float('inf')
        
        # Total wait = queue wait + backlog effect
        total_wait = mgc_results['W_q'] + backlog_clear_time
        
        return {
            'provider': provider_data.get('provider', 'Unknown'),
            'type': provider_type,
            'theatres': theatres,
            'lambda': lambda_rate,
            'mu': mu_per_server,
            'weekly_capacity': weekly_capacity,
            'utilization': mgc_results['utilization'],
            'cv': cv,
            'queue_wait': mgc_results['W_q'],
            'backlog_size': backlog_size,
            'backlog_clear_time': backlog_clear_time,
            'total_wait': total_wait
        }
    
    def analyze_consolidated_system(self, all_providers):
        """Analysing consolidated queue system"""
        # Aggregate parameters
        total_lambda = 0
        total_capacity = 0
        total_theatres = 0
        total_backlog = 0
        weighted_cv_sum = 0
        
        for provider in all_providers:
            if provider:
                total_lambda += provider['lambda']
                total_capacity += provider['weekly_capacity']
                total_theatres += provider['theatres']
                total_backlog += provider['backlog_size']
                weighted_cv_sum += provider['cv'] * provider['backlog_size']
        
        if total_backlog > 0:
            weighted_cv = weighted_cv_sum / total_backlog
        else:
            weighted_cv = 0.7
        
        # Average service rate per server
        mu_avg = total_capacity / total_theatres if total_theatres > 0 else 0
        
        mmc_results = self.erlang_c_formula(total_lambda, mu_avg, total_theatres)    # M/M/c for consolidated
        
        mgc_results = self.mgc_correction(mmc_results, weighted_cv) # M/G/c correction
        
        # Backlog clearance
        if total_capacity > total_lambda:
            efficiency_factor = 0.2 # derived from DES efficiency
            backlog_clear_time = efficiency_factor * (total_backlog / max(total_capacity - total_lambda, 1e-3))
        else:
            backlog_clear_time = float('inf')
        
        total_wait = mgc_results['W_q'] + backlog_clear_time
        
        return {
            'total_theatres': total_theatres,
            'total_lambda': total_lambda,
            'total_capacity': total_capacity,
            'utilization': mgc_results['utilization'],
            'weighted_cv': weighted_cv,
            'queue_wait': mgc_results['W_q'],
            'total_backlog': total_backlog,
            'backlog_clear_time': backlog_clear_time,
            'total_wait': total_wait
        }

def run_analysis():
    """Run analysis with realistic wait times"""
    start_time = time.time()
    # Load data
    print("\nLoading data")
    try:
        patient_data = pd.read_csv('new_data_ref_dates.csv')
        provider_data = pd.read_csv('provider_capacity_data.csv')
        print(f" Loaded {len(patient_data)} patient records")
        print(f" Loaded {len(provider_data)} provider records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Main providers
    main_providers = [
        'NORTH WEST ANGLIA NHS FOUNDATION TRUST',
        'CAMBRIDGE UNIVERSITY HOSPITALS NHS FOUNDATION TRUST',
        'ANGLIA COMMUNITY EYE SERVICE LTD',
        'SPAMEDICA PETERBOROUGH',
        'SPAMEDICA BEDFORD',
        'FITZWILLIAM HOSPITAL'
    ]
    
    # Filter data
    provider_data_main = provider_data[provider_data['provider'].isin(main_providers)].copy()
    
    # Add parameters
    #theatre counts assumed based on historical data in provider_capacity_data.csv
    theatre_counts = {
        'NORTH WEST ANGLIA NHS FOUNDATION TRUST': 3,
        'CAMBRIDGE UNIVERSITY HOSPITALS NHS FOUNDATION TRUST': 3,
        'ANGLIA COMMUNITY EYE SERVICE LTD': 4,
        'SPAMEDICA PETERBOROUGH': 3,
        'SPAMEDICA BEDFORD': 1,
        'FITZWILLIAM HOSPITAL': 1
    }
    
    provider_data_main['theatres'] = provider_data_main['provider'].map(theatre_counts).fillna(1)
    
    # PART 1: M/G/c Analysis
    print("PART 1: M/G/c QUEUING ANALYSIS")    
    mgc_analyzer = MGcQueueClass()
    
    print(f"\nService rates increased by {(mgc_analyzer.service_rate_multiplier-1)*100:.0f}% for realistic wait times")
    print(f"Backlog reduced to {mgc_analyzer.assumed_backlog_months['NHS']} months (NHS) and {mgc_analyzer.assumed_backlog_months['Independent']} months (Independent)")
    
    # Analyze separate queues
    print("\nSEPARATE QUEUES ANALYSIS:")
    
    separate_results = []
    
    for _, provider in provider_data_main.iterrows():
        provider_name = provider['provider']
        provider_type = 'NHS' if 'NHS' in provider['provider_type'] else 'Independent'
        
        result = mgc_analyzer.run_analysis(provider, provider_type)
        
        if result:
            separate_results.append(result)
            
            print(f"\n{provider_name[:40]}:")
            print(f"  Type: {provider_type}")
            print(f"  Theatres: {result['theatres']}")
            print(f"  Service rate (μ) per theatre: {result['mu']:.2f} patients/week")
            print(f"  Arrival rate (λ): {result['lambda']:.2f} patients/week")
            print(f"  Utilization: {result['utilization']*100:.1f}%")
            print(f"  CV: {result['cv']}")
            print(f"  Backlog: {result['backlog_size']} patients")
            print(f"  Queue wait (M/G/c): {result['queue_wait']:.3f} weeks")
            print(f"  Backlog clear time: {result['backlog_clear_time']:.1f} weeks")
            print(f"  Total wait: {result['total_wait']:.1f} weeks")
    
    # Analyze consolidated system
    print("\n\n CONSOLIDATED QUEUE ANALYSIS:")
    
    consolidated_result = mgc_analyzer.analyze_consolidated_system(separate_results)
    
    print(f"  Total theatres: {consolidated_result['total_theatres']}")
    print(f"  Total arrival rate (λ): {consolidated_result['total_lambda']:.2f} patients/week")
    print(f"  Total capacity: {consolidated_result['total_capacity']:.2f} patients/week")
    print(f"  System utilization: {consolidated_result['utilization']*100:.1f}%")
    print(f"  Weighted CV: {consolidated_result['weighted_cv']:.3f}")
    print(f"  Total backlog: {consolidated_result['total_backlog']} patients")
    print(f"  Queue wait (M/G/c): {consolidated_result['queue_wait']:.3f} weeks")
    print(f"  Backlog clear time: {consolidated_result['backlog_clear_time']:.1f} weeks")
    print(f"  Total wait: {consolidated_result['total_wait']:.1f} weeks")
    
    # Calculate M/G/c improvement
    avg_separate_wait = np.mean([r['total_wait'] for r in separate_results])
    mgc_improvement = (avg_separate_wait - consolidated_result['total_wait']) / avg_separate_wait * 100
    
    print(f"\n  COMPARISON:")
    print(f"  Separate queues average: {avg_separate_wait:.1f} weeks")
    print(f"  Consolidated queue: {consolidated_result['total_wait']:.1f} weeks")
    print(f"  M/G/c Improvement: {mgc_improvement:.1f}%")
    

if __name__ == "__main__":
    run_analysis()
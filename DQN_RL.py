import numpy as np
import pandas as pd
from collections import deque, defaultdict
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class Config:
    # Environment
    n_providers = 6
    state_dim = 27  # 6 providers × 4 features + 3 global features
    episodes = 1000 
    
    # Initial backlogs
    nhs_backlog_weeks = 40
    independent_backlog_weeks = 26
    
    # DQN hyperparameters 
    lr = 0.001  # learning rate α
    gamma = 0.95  # discount factor γ
    epsilon_start = 1.0  # initial ε
    epsilon_decay = 0.995  # decay rate
    epsilon_min = 0.01  # min ε
    
    batch_size = 64  # batch size
    buffer_size = 10000  # replay buffer D capacity
    target_update_freq = 100  # update θ− every 100 steps
    
    # Network architecture [27→256→128→64→6]
    hidden_sizes = [256, 128, 64]

#provider class to manage queues and states
@dataclass
class Provider:
    """Provider with backlog initialization"""
    name: str
    provider_type: str
    capacity_per_week: float
    
    queue: deque = field(default_factory=lambda: deque())
    
    def post_init(self):
        # Initialising backlogs as per algorithm
        if 'NHS' in self.provider_type:
            backlog_weeks = 40
        else:
            backlog_weeks = 26
            
        # Creating backlog patients
        backlog_size = int(backlog_weeks * self.capacity_per_week)
        for i in range(backlog_size):
            self.queue.append({'id': f'B{i}', 'wait_time': backlog_weeks})
    
    @property
    def current_wait(self):
        # W_total = W_q + B₀/μ_effective
        # Current wait time based on queue length and capacity
        if self.capacity_per_week == 0:
            return 60.0
        return min(len(self.queue) / self.capacity_per_week, 60.0)
    
    @property
    def utilization(self):
        # Utilization based on queue length and capacity
        target = 0.93 if 'NHS' in self.provider_type else 0.88
        pressure = min(len(self.queue) / (self.capacity_per_week * 10), 1.0)
        return target * (0.8 + 0.4 * pressure)

#PTL Environment
class PTLEnvironment(gym.Env):   
    def __init__(self, config, provider_data):
        self.config = config
        self.providers = self.init_providers(provider_data)
        
        # State space S (27-dim) 
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(config.state_dim,), dtype=np.float32
        )
        
        # Action space A (6 providers)
        self.action_space = spaces.Discrete(config.n_providers)
        
        self.step_count = 0
        self.max_steps = 200  # Terminal condition
        
    def init_providers(self, data):
        # Initialising providers from data
        providers = []
        for _, row in data.iterrows():
            p = Provider(
                name=row['provider'],
                provider_type='NHS' if 'NHS' in row['provider'] else 'Independent',
                capacity_per_week=row['weekly_capacity']
            )
            providers.append(p)
        return providers
    
    def reset(self, seed=None):
        # Reset environment to initial state with backlogs
        super().reset(seed=seed)
        self.step_count = 0
        
        # Re-initialize providers with backlogs
        for p in self.providers:
            p.queue.clear()
            p.post_init()  # Re-initialize backlog
            
        return self.get_state(), {}
    
    def get_state(self):
        # Constructing state vector (27-dim)   
        state = []
        
        # For each provider (6 providers × 4 features = 24 dims)
        for p in self.providers:
            # Normalised wait time
            wait_norm = p.current_wait / 60.0
            # utilisation
            util = p.utilization
            # Queue size normalised
            queue_norm = min(len(p.queue) / 500, 1.0)
            # Provider type (NHS=1, Independent=0)
            provider_type = 1.0 if 'NHS' in p.provider_type else 0.0
            
            state.extend([wait_norm, util, queue_norm, provider_type])
        
        # Global features (3 dimensions)
        avg_wait = np.mean([p.current_wait for p in self.providers])
        avg_util = np.mean([p.utilization for p in self.providers])
        time_norm = self.step_count / self.max_steps
        
        state.extend([avg_wait/60.0, avg_util, time_norm])
        
        return np.array(state[:27], dtype=np.float32)  
    
    def step(self, action):
        #patient allocation to chosen provider
        provider = self.providers[action]
        
        # Adding patient to chosen provider
        patient = {'id': f'P{self.step_count}', 'wait_time': 0}
        provider.queue.append(patient)
        
        # Calculating reward r
        reward = self.calculate_reward(provider)
        
        # Process queues
        self.process_queues()
        
        # Update step count
        self.step_count += 1
        
        # Checking the terminal condition
        terminal = self.step_count >= self.max_steps
        
        # Get the next state s'
        next_state = self.get_state()
        
        info = {'avg_wait': np.mean([p.current_wait for p in self.providers])}
        
        return next_state, reward, terminal, False, info
    
    def calculate_reward(self, provider):
        # Reward function r
        # aim: minimise wait times and maintain good utilization
        # Negative reward for wait time
        r = -provider.current_wait / 10
        
        # Penalty for poor utilization
        if provider.utilization > 0.95:
            r -= 2.0  # Overloaded
        elif provider.utilization < 0.85:
            r -= 1.0  # Underutilized
        else:
            r += 1.0  # Good utilization
            
        return r
    
    def process_queues(self):
        # Process each providers queue based on capacity
        for p in self.providers:
            serve_count = int(p.capacity_per_week / self.max_steps)
            for _ in range(min(serve_count, len(p.queue))):
                if p.queue:
                    p.queue.popleft()

# Q network architecture
class QNetwork(nn.Module):
    # Neural network for Q-value approximation    
    def __init__(self, input_dim=27, output_dim=6, hidden_sizes=[256, 128, 64]):
        super(QNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers as per architecture
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# DQN Agent
class DQNAgent:
    """DQN Agent implementing Algorithm 3.4"""
    
    def __init__(self, config):
        self.config = config
        
        # Initializing Q-network Q(s,a;θ)
        self.q_network = QNetwork(
            input_dim=config.state_dim,
            output_dim=config.n_providers,
            hidden_sizes=config.hidden_sizes
        )
        
        # Initializing target network 
        self.target_network = QNetwork(
            input_dim=config.state_dim,
            output_dim=config.n_providers,
            hidden_sizes=config.hidden_sizes
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer with learning rate 
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        
        # Replay buffer D
        self.replay_buffer = deque(maxlen=config.buffer_size)
        
        # Initialising exploration ε
        self.epsilon = config.epsilon_start
        
        self.steps = 0
    
    def select_action(self, state):
        # ε-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, self.config.n_providers - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state))
                return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, terminal):
        # StorING experience tuple in replay buffer D
        self.replay_buffer.append((state, action, reward, next_state, terminal))
    
    def train_step(self):
        # Training step using mini-batch from replay buffer
        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.config.batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        terminals = torch.FloatTensor([e[4] for e in batch])
        
        # Calculate targets
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            targets = rewards + (1 - terminals) * self.config.gamma * next_q
        
        # Calculate loss (MSE)
        loss = nn.MSELoss()(current_q.squeeze(), targets)
        
        # Gradient descent update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        
        if self.steps % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration: ε = max(0.01, ε × 0.995)"""
        self.epsilon = max(self.config.epsilon_min, 
                          self.epsilon * self.config.epsilon_decay)

# dqn training function
def train_dqn(env, config):
    """
    Input: State space S (27-dim), action space A (6 providers), 
           episodes=1000, batch_size=64, buffer_size=10000, 
           learning_rate=0.001, γ=0.95
    Output: Trained Q-network Q(s,a;θ)
    """
    
    # Initialize agent
    agent = DQNAgent(config)
    
    episode_rewards = []
    episode_waits = []
    
    print("Starting DQN Training (Algorithm 3.4)")
    print(f"State space: {config.state_dim}-dim")
    print(f"Action space: {config.n_providers} providers")
    print(f"Episodes: {config.episodes}")
    print("-" * 60)
    
    for episode in range(config.episodes):
        # s ← reset environment with backlogs
        state, _ = env.reset()
        
        episode_reward = 0
        waits = []
        terminal = False
        
        while not terminal:
            # ε-greedy action selection
            action = agent.select_action(state)
            
            # Execute allocation decision
            next_state, reward, terminal, _, info = env.step(action)
            
            # Store (s,a,r,s') in D
            agent.store_experience(state, action, reward, next_state, terminal)
            
            # Training step (if sufficient experience)
            agent.train_step()
            
            # Track metrics
            episode_reward += reward
            waits.append(info['avg_wait'])
            
            # s ← s'
            state = next_state
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_waits.append(np.mean(waits))
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_wait = np.mean(episode_waits[-100:])
            print(f"Episode {episode+1}/{config.episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Wait: {avg_wait:.1f} weeks")
            print(f"  Epsilon: {agent.epsilon:.4f}")
    
    print("\nTraining Complete!")
    print(f"Final ε: {agent.epsilon:.4f}")
    
    # return Q-network with parameters θ
    return agent, episode_rewards, episode_waits

#Evaluation function
def evaluate_policies(env, trained_agent=None):
        
    n_eval_episodes = 10
    # 1. Random Policy (baseline)
    print("Random Allocation")
    env.reset()
    random_waits = []
    
    for _ in range(n_eval_episodes):
        state, _ = env.reset()
        episode_waits = []
        done = False
        
        while not done:
            action = random.randint(0, env.action_space.n - 1)
            _, _, done, _, info = env.step(action)
            episode_waits.append(info['avg_wait'])
        
        random_waits.append(np.mean(episode_waits))
    
    print(f"Average Wait: {np.mean(random_waits):.1f} weeks")
    
    # 2. DQN Policy
    if trained_agent:
        print("\n DQN Policy")
        dqn_waits = []
        
        for _ in range(n_eval_episodes):
            state, _ = env.reset()
            episode_waits = []
            done = False
            
            # Use trained Q-network 
            trained_agent.epsilon = 0  # Greedy policy for evaluation
            
            while not done:
                action = trained_agent.select_action(state)
                state, _, done, _, info = env.step(action)
                episode_waits.append(info['avg_wait'])
            
            dqn_waits.append(np.mean(episode_waits))
        
        print(f"   Average Wait: {np.mean(dqn_waits):.1f} weeks")
        
        # Show improvement
        improvement = (np.mean(random_waits) - np.mean(dqn_waits)) / np.mean(random_waits) * 100
        print(f"\nDQN Improvement vs Random: {improvement:.1f}%")

def main():
    # Top provider data
    provider_data = pd.DataFrame({
        'provider': [
            'NORTH WEST ANGLIA NHS',
            'CAMBRIDGE UNIVERSITY NHS',
            'ANGLIA COMMUNITY EYE',
            'SPAMEDICA PETERBOROUGH', 
            'SPAMEDICA BEDFORD',
            'FITZWILLIAM HOSPITAL'
        ],
        'weekly_capacity': [150, 175, 200, 125, 75, 50]
    })
    
    # Initialize environment
    config = Config()
    env = PTLEnvironment(config, provider_data)
    
    print(f"\nInitialized {config.n_providers} providers")
    print(f"NHS providers: 40-week initial backlog")
    print(f"Independent providers: 26-week initial backlog")
    print(f"\nNetwork Architecture: {config.state_dim}→{config.hidden_sizes[0]}→{config.hidden_sizes[1]}→{config.hidden_sizes[2]}→{config.n_providers}")
    print(f"Hyperparameters: γ={config.gamma}, α={config.lr}, ε={config.epsilon_start}")
    print(f"Batch size: {config.batch_size}, Buffer: {config.buffer_size}")
    
    # Training
    agent, rewards, waits = train_dqn(env, config)
    
    # Evaluation
    evaluate_policies(env, agent)    
    return agent, rewards, waits

if __name__ == "__main__":
    agent, rewards, waits = main()
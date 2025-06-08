from typing import List, Dict, Tuple
import numpy as np

class UCBCalculator:
    def __init__(
        self, 
        all_clients: List[int], 
        num_models: int,
        data_ratio: Dict[int, List[float]],
        gamma: float = 0.9
    ):
        self.all_clients = all_clients
        self.M = num_models
        self.gamma = gamma
        self.data_ratio = data_ratio
        
        self.history = {
            c: {
                m: {"losses": [], "selected_rounds": []}
                for m in range(num_models)
            }
            for c in all_clients
        }

    def update(self, client: int, model: int, loss: float, current_round: int) -> None:
        self.history[client][model]["losses"].append(loss)
        self.history[client][model]["selected_rounds"].append(current_round)

    def compute_scores(self, current_round: int) -> Dict[int, List[float]]:
        client_scores = {}
        for c in self.all_clients:
            scores = []
            for m in range(self.M):
                losses = self.history[c][m]["losses"]
                selected_rounds = self.history[c][m]["selected_rounds"]
                
                # Calculate weighted loss
                L = sum(self.gamma ** (current_round - r) * loss 
                       for loss, r in zip(losses, selected_rounds))
                
                # Calculate weighted selection count
                N = sum(self.gamma ** (current_round - r) for r in selected_rounds)
                
                # Safely calculate exploration term
                total_exploration = sum(self.gamma ** (current_round - r) 
                                   for r in range(current_round))
                U = 0.0
                if N == 0:
                    U = np.inf
                else:
                    log_arg = max(total_exploration, 1e-6)
                    log_val = np.log(log_arg + 1e-6)
                    U = np.sqrt(2 * max(log_val, 0) / max(N, 1e-6))
                
                # Calculate final score
                p = self.data_ratio[c][m]
                score = p * (L / max(N, 1e-6) + U)
                scores.append(round(score, 4))
            
            client_scores[c] = scores
        return client_scores

def is_pareto_efficient(scores: Dict[int, List[float]]) -> List[int]:
    clients = list(scores.keys())
    num_clients = len(clients)
    is_efficient = np.ones(num_clients, dtype=bool)
    
    for i in range(num_clients):
        if not is_efficient[i]:
            continue
            
        for j in range(num_clients):
            if i == j or not is_efficient[j]:
                continue
                
            # Handle inf value comparison
            dominated = True
            has_advantage = False
            for m in range(len(scores[clients[i]])):
                si = scores[clients[i]][m]
                sj = scores[clients[j]][m]
                
                if np.isinf(si) and np.isinf(sj):
                    continue  # Skip comparison if both are inf
                elif sj < si:
                    dominated = False
                    break
                elif sj > si:
                    has_advantage = True
                    
            if dominated and has_advantage:
                is_efficient[i] = False
                break
                
    return [clients[i] for i in range(num_clients) if is_efficient[i]]

def pareto_multi_ucb(
    client_scores: Dict[int, List[float]], 
    all_clients: List[int], 
    K: int, 
    current_round: int, 
    M: int
    ) -> Tuple[List[int], List[int]]:
    # Force selection of all clients in the first round
    # if current_round == 0:
    #     selected_clients = all_clients[:K]
    #     assigned_models = [np.random.choice(np.flatnonzero(
    #         scores == np.max(scores)))  # Randomly select the highest scoring model
    #         for scores in client_scores.values()]
    #     return selected_clients, assigned_models[:K]
    
    if current_round <= 2:
        return [0,1,2], [[0,1,2],[1,2,0],[2,0,1]][current_round]
    
    # Get Pareto-efficient clients
    pareto_clients = is_pareto_efficient(client_scores)
    
    # Ensure at least K clients are selected
    selected = []
    remaining = K
    
    # Phase 1: Select clients 
    if len(pareto_clients) > 0:
        select_num = min(len(pareto_clients), remaining)
        selected += list(np.random.choice(pareto_clients, select_num, replace=False))
        remaining -= select_num
    
    # Phase 2: If not enough, select from remaining clients
    if remaining > 0:
        non_pareto = [c for c in all_clients if c not in pareto_clients]
        selected += list(np.random.choice(non_pareto, remaining, replace=False))
    
    # Model assignment logic
    assigned_models = []
    for c in selected:
        valid_scores = [s if not np.isinf(s) else -np.inf for s in client_scores[c]]
        best_model = np.argmax(valid_scores)
        assigned_models.append(best_model)

    # sort the selected clients and assigned models
    selected = sorted(selected)
    assigned_models = [assigned_models[selected.index(c)] for c in selected]
    
    return selected, assigned_models

if __name__ == "__main__":
    # Configuration parameters
    num_clients = 3
    M = 3
    K = 3
    gamma = 0.9
    model_mapping = {0: "A", 1: "B", 2: "C"}

    # Generate data distribution (ensure each client has data for at least one model)
    np.random.seed(42)
    data_ratio = {
        c: [max(v, 0.1) for v in np.random.dirichlet(np.ones(M))]  # Ensure minimum data ratio
        for c in range(num_clients)
    }
    
    # Initialize system
    ucb_calculator = UCBCalculator(
        all_clients=list(range(num_clients)),
        num_models=M,
        data_ratio=data_ratio,
        gamma=gamma
    )
    
    client_model_counts = {c: {m: 0 for m in range(M)} for c in range(num_clients)}

    # Training loop
    for current_round in range(11):
        print(f"\n=== Round {current_round} ===")

        # Compute scores
        client_scores = ucb_calculator.compute_scores(current_round)

        # Select clients
        selected_clients, assigned_models = pareto_multi_ucb(
            client_scores=client_scores,
            all_clients=list(range(num_clients)),
            K=K,
            current_round=current_round,
            M=M
        )

        # Simulate training
        for c, m in zip(selected_clients, assigned_models):
            loss = np.random.uniform(0.1, 1.0)
            print(c,m)
            ucb_calculator.update(c, m, loss, current_round)
            client_model_counts[c][m] += 1

        # Print results
        print(f"Selected Clients: {selected_clients}")
        print(f"Assigned Models:  {[model_mapping[m] for m in assigned_models]}")
        print("Client Scores:")
        for c in selected_clients:
            scores = client_scores[c]
            print(f"  Client {c}: " + 
                  ", ".join([f"Model_{model_mapping[m]}={s:.4f}" 
                           for m, s in enumerate(scores)]))

    # Training summary
    print("\n=== Training Summary ===")
    for c in range(num_clients):
        print(f"Client {c}:")
        for m in range(M):
            print(f"  Model {model_mapping[m]}: {client_model_counts[c][m]} times")
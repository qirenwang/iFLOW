from typing import List, Dict, Tuple
import numpy as np

class UCBCalculator:
    def __init__(
        self, 
        all_clients: List[int], 
        num_models: int,
        data_ratio: Dict[int, List[float]],  # client_id -> [p_model1, p_model2, ...]
        gamma: float = 0.9
    ):
        """
        Initialize the UCB score calculator.

        Args:
            all_clients (List[int]): List of all client IDs.
            num_models (int): Total number of models.
            data_ratio (Dict[int, List[float]]): Data ratio of clients on each model.
            gamma (float): Discount factor for historical losses (default 0.9).
        """
        self.all_clients = all_clients
        self.M = num_models
        self.gamma = gamma
        self.data_ratio = data_ratio
        
        # Initialize history: client -> model -> [losses], [selected_rounds]
        self.history = {
            c: {
                m: {"losses": [], "selected_rounds": []}
                for m in range(num_models)
            }
            for c in all_clients
        }

    def update(
        self, 
        client: int, 
        model: int, 
        loss: float, 
        current_round: int
    ) -> None:
        """
        Update the training history of the client on the specified model.

        Args:
            client (int): Client ID.
            model (int): Model ID.
            loss (float): Local training loss of the current round.
            current_round (int): Current training round.
        """
        self.history[client][model]["losses"].append(loss)
        self.history[client][model]["selected_rounds"].append(current_round)

    def compute_scores(self, current_round: int) -> Dict[int, List[float]]:
        """
        Compute the UCB scores of all clients on all models.

        Args:
            current_round (int): Current training round (starting from 0).

        Returns:
            Dict[int, List[float]]: Mapping of client ID to UCB scores of each model.
        """
        client_scores = {}
        for c in self.all_clients:
            scores = []
            for m in range(self.M):
                # Extract historical losses and selection rounds
                losses = self.history[c][m]["losses"]
                selected_rounds = self.history[c][m]["selected_rounds"]
                
                # Compute weighted loss L_t(k,i)
                L = 0.0
                for loss, r in zip(losses, selected_rounds):
                    decay = self.gamma ** (current_round - r)
                    L += decay * loss
                
                # Compute weighted selection count N_t(k,i)
                N = sum(self.gamma ** (current_round - r) for r in selected_rounds)
                
                # Compute exploration term U_t(k,i)
                total_exploration = sum(self.gamma ** (current_round - r) for r in range(current_round))
                if N == 0:
                    U = np.inf  # Never selected, prioritize exploration
                else:
                    U = np.sqrt(2 * np.log(total_exploration + 1e-6) / N)
                
                # Compute final UCB score
                p = self.data_ratio[c][m]
                score = p * (L / (N + 1e-6) + U)
                scores.append(round(score, 4))
            
            client_scores[c] = scores
        return client_scores
    
def ranklist_multi_ucb(
    client_scores: Dict[int, List[float]], 
    all_clients: List[int], 
    K: int, 
    current_round: int, 
    M: int
    ) -> Tuple[List[int], List[int]]:
    """
    Implement the Ranklist-Multi-UCB strategy for client selection and model assignment.

    Args:
        client_scores (Dict[int, List[float]]): 
            Mapping of client ID to UCB scores of each model, format {client_id: [score_model1, score_model2, ...]}.
        all_clients (List[int]): 
            List of all selectable client IDs.
        K (int): 
            Number of clients to select each round.
        current_round (int): 
            Current training round (starting from 0).
        M (int): 
            Total number of models.

    Returns:
        Tuple[List[int], List[int]]: 
            List of selected client IDs and list of assigned model IDs, e.g., ([1,3,5], [2,1,2]) means:
            Client 1 trains model 2, client 3 trains model 1, client 5 trains model 2.
    """
    # Initialize data structures
    selected_clients = []
    assigned_models = []
    model_ranklists = {m: [] for m in range(M)}  # Client ranking list for each model

    # Step 1: Generate client ranking for each model (sorted by UCB score in descending order)
    for m in range(M):
        # Extract client scores on model m and sort
        sorted_clients = sorted(
            all_clients,
            key=lambda c: client_scores[c][m], 
            reverse=True
        )
        model_ranklists[m] = sorted_clients

    # Step 2: Determine the starting model (round-robin logic)
    start_model = (current_round % M)  # Model index starts from 0
    model_order = [(start_model + i) % M for i in range(M)]  # Round-robin order

    # Step 3: Round-robin client selection
    count = 0
    while len(selected_clients) < K:
        # Current round-robin model
        current_model = model_order[count % M]
        # Select the first unselected client from the ranking list of the model
        for client in model_ranklists[current_model]:
            if client not in selected_clients:
                selected_clients.append(client)
                assigned_models.append(current_model)
                break  # Break after selection, process the next client
        count += 1

        # If all clients are selected but not reaching K, terminate early (should be avoided in practice)
        if count > M * len(all_clients):
            break

    # sort the selected clients and assigned models
    selected_clients = sorted(selected_clients)
    assigned_models = [assigned_models[selected_clients.index(c)] for c in selected_clients]

    return selected_clients, assigned_models



if __name__ == "__main__":
    # Parameter configuration
    num_clients = 3
    M = 3
    K = 3
    gamma = 0.9
    model_mapping = {0: "A", 1: "B", 2: "C"}

    # Generate fake data ratio (assuming different data distribution for each client)
    np.random.seed(42)
    data_ratio = {
        c: list(np.random.dirichlet(np.ones(M)))  # Dirichlet distribution ensures sum to 1
        for c in range(num_clients)
    }
    # Initialize UCB calculator
    ucb_calculator = UCBCalculator(
        all_clients=list(range(num_clients)),
        num_models=M,
        data_ratio=data_ratio,
        gamma=gamma
    )
    # Initialize a counter for each client and model
    client_model_counts = {c: {m: 0 for m in range(M)} for c in range(num_clients)}

    # Simulate multiple training rounds
    for current_round in range(11):  # Rounds 0 to 10
        print(f"\n=== Round {current_round} ===")

        # 1. Compute UCB scores for the current round
        client_scores = ucb_calculator.compute_scores(current_round)

        # 2. Call Ranklist-Multi-UCB to select clients and models
        selected_clients, assigned_models = ranklist_multi_ucb(
            client_scores=client_scores,
            all_clients=list(range(num_clients)),
            K=K,
            current_round=current_round,
            M=M
        )

        # 3. Simulate client training and update history (randomly generate losses)
        for c, m in zip(selected_clients, assigned_models):
            # Randomly generate local loss (assuming loss between 0.1 and 1.0)
            loss = np.random.uniform(0.1, 1.0)
            ucb_calculator.update(c, m, loss, current_round)
            client_model_counts[c][m] += 1  # Update the counter

        # Print results
        print(f"Selected Clients: {selected_clients}")

        assigned_models_mapping = [model_mapping[m] for m in assigned_models]
        print(f"Assigned Models:  {assigned_models_mapping}")
        
        print("Client Scores:")
        for c in selected_clients:
            scores = client_scores[c]
            print(f"  Client {c}: Model_A={scores[0]:.4f}, Model_B={scores[1]:.4f}, Model_C={scores[2]:.4f}")

    # Print summary of training counts
    print("\n=== Training Summary ===")
    for c in range(num_clients):
        print(f"Client {c}:")
        for m in range(M):
            print(f"  Model {m}: {client_model_counts[c][m]} times")

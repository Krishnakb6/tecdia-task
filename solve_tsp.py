import numpy as np
import time
import sys
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# --- Configuration ---
MATRIX_FILE = 'similarity_matrix.npy'
FRAME_COUNT = 300

print(f"--- SCRIPT STARTED (Google OR-Tools TSP) ---")

# 1. Load the Cost Matrix
print(f"Loading matrix from {MATRIX_FILE}...")
try:
    # Load the SIMILARITY matrix first
    similarity_matrix = np.load(MATRIX_FILE)
    if similarity_matrix.shape != (FRAME_COUNT, FRAME_COUNT):
        raise ValueError(f"Matrix shape is {similarity_matrix.shape}")
    
    # Convert to COST matrix
    # We must use INTEGERS for ortools, so we scale by a large number
    print("Converting to integer cost matrix...")
    cost_matrix = (1.0 - similarity_matrix) * 1_000_000
    cost_matrix = cost_matrix.astype(np.int64)
    
    # Set the diagonal (cost to self) to a very high number
    np.fill_diagonal(cost_matrix, 999_999_999) # A large "infinity"
    
    print("--- STEP 1: Cost matrix loaded successfully.")

except Exception as e:
    print(f"Error loading matrix: {e}")
    exit()


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = cost_matrix
    data['num_vehicles'] = 1  # We are finding one single path
    data['depot'] = 0         # We will start our path at Frame 0
    return data

def get_solution(manager, routing, solution):
    """Gets the final path from the solver solution."""
    path = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        path.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    # Add the last node
    path.append(manager.IndexToNode(index))
    return path


# 2. Run the OR-Tools TSP Solver
print("--- STEP 2: Starting Google OR-Tools TSP solver...")
start_time = time.time()

# Create the data model
data = create_data_model()

# Create the routing index manager
manager = pywrapcp.RoutingIndexManager(
    len(data['distance_matrix']), # Number of "cities" (300)
    data['num_vehicles'],        # Number of "salesmen" (1)
    data['depot']                # The starting city (Frame 0)
)

# Create the Routing Model
routing = pywrapcp.RoutingModel(manager)

def distance_callback(from_index, to_index):
    """Returns the distance between two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

# Define the cost of travel
transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Setting first solution heuristic (a good starting guess)
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
search_parameters.time_limit.seconds = 30 # Max time to search

# Solve the problem
solution = routing.SolveWithParameters(search_parameters)
end_time = time.time()

# 3. Process and Save the Result
if solution:
    print(f"--- STEP 3: Solver FINISHED in {end_time - start_time:.2f} seconds.")
    
    # Get the path (it will be a loop starting and ending at 0)
    path = get_solution(manager, routing, solution)
    
    # The path is a loop (e.g., [0, 15, 32, ..., 112, 0])
    # We must find the "seam" (worst link)
    print("--- STEP 4: Finding the 'seam' (worst link)...")
    
    worst_cost = -1
    worst_link_index = -1
    
    for i in range(FRAME_COUNT):
        # We use [:-1] to ignore the last '0' in the loop path
        frame_a = path[i]
        frame_b = path[i + 1] # The next frame in the path
        
        cost = cost_matrix[frame_a, frame_b]
        if cost > worst_cost:
            worst_cost = cost
            worst_link_index = i
            
    # The worst link is from path[worst_link_index] -> path[worst_link_index + 1]
    # So, the TRUE path starts at path[worst_link_index + 1]
    start_node = path[worst_link_index + 1]
    
    # Re-order the path to start at the correct frame
    # This is a bit complex, but we find the start_node's
    # position in the path and "rotate" the list
    start_index_in_path = path.index(start_node)
    
    # Use [:-1] to drop the looping "0" at the end
    final_ordered_path = path[start_index_in_path:-1] + path[:start_index_in_path]

    print(f"--- Found start frame: {final_ordered_path[0]} ---")
    
    # 6. Save the Result
    print(f"Optimal frame order (first 15 frames): {final_ordered_path[:15]}...")
    np.save('optimal_order.npy', final_ordered_path)
    print(f"\nSuccessfully saved the optimal frame order to 'optimal_order.npy'")
else:
    print('No solution found!')
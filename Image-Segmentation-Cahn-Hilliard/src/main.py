import argparse
import os
import sys

# Add the project root to the Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import experiments

def main():
    parser = argparse.ArgumentParser(description="Run Cahn-Hilliard image segmentation experiments.")
    parser.add_argument(
        '--experiment',
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="The experiment number to run (1-4)."
    )
    args = parser.parse_args()

    # Define project structure paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    results_base_dir = os.path.join(base_dir, 'results')

    # Define the specific results directory for this experiment
    exp_results_dir = os.path.join(results_base_dir, f'experiment_{args.experiment}')
    os.makedirs(exp_results_dir, exist_ok=True)

    # Check if data directory exists
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found at '{data_dir}'")
        print("Please create it and add the required images.")
        sys.exit(1)

    # A mapping from experiment number to function
    experiment_functions = {
        1: experiments.run_experiment_1,
        2: experiments.run_experiment_2,
        3: experiments.run_experiment_3,
        4: experiments.run_experiment_4,
    }

    # Get the function for the chosen experiment
    run_experiment = experiment_functions.get(args.experiment)
    
    if run_experiment:
        print(f"--- Running Experiment {args.experiment} ---")
        run_experiment(data_dir, exp_results_dir)
        print(f"plots are saved to: {os.path.relpath(exp_results_dir)}")
    else:
        # This case should not be reached due to argparse choices
        print(f"Error: Experiment {args.experiment} is not defined.")

if __name__ == "__main__":
    main()
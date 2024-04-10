import argparse
import pandas as pd
from model.model import RecommenderSystemModel
from utils import plot_agent_vector

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mesa model.")
    parser.add_argument("--n_users", type=int, default=100, help="Number of users.")
    parser.add_argument("--dummy", type=bool, default=False, help="Use dummy data. Needs to be presaved.")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps.")
    parser.add_argument("--plot_agent", type=int, default=0, help="Return plots of agent.")
    parser.add_argument("--priority", type=str | None, default=None, help="Hidden item priority. Available options: None, 'random' or item category")
    args = parser.parse_args()
    rec_sys = RecommenderSystemModel(n_users=args.n_users, dummy=args.dummy, steps=args.steps, priority=args.priority)
    rec_sys.run_model()
    results_df = rec_sys.get_processed_df()
    if args.plot_agent:
        plot_agent_vector(results_df, args.plot_agent)

if __name__ == "__main__":
    main()
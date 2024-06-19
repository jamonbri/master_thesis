import mesa
import pandas as pd
import numpy as np
from model.agents import ItemAgent, UserAgent
from data.data_preparation import get_model_df, get_users_df, get_items_df, get_categories
from data.results import Results


def get_vector(agent: mesa.Agent) -> np.array:
    """
    Helper method to get user agent vector for data collection
    """
    if isinstance(agent, UserAgent):
        return agent.vector.copy()
    return None

def get_books_consumed(agent: mesa.Agent) -> list[int]:
    """
    """
    if isinstance(agent, UserAgent):
        return agent.books_consumed.copy()
    return None

def get_agent_type(agent: mesa.Agent) -> str:
    """
    Helper method to get agent model type
    """
    return agent.__class__.__name__

def get_item_n_read(agent: mesa.Agent) -> int:
    """
    Helper method to get count of item reads
    """
    if isinstance(agent, ItemAgent):
        return agent.n_read
    return None

def get_item_n_reviews(agent: mesa.Agent) -> int:
    """
    Helper method to get count of item reviews
    """
    if isinstance(agent, ItemAgent):
        return agent.n_reviews
    return None

def get_item_mean_rating(agent: mesa.Agent) -> int:
    """
    Helper method to get mean item rating
    """
    if isinstance(agent, ItemAgent):
        return agent.mean_rating
    return None


class RecommenderSystemModel(mesa.Model):
    """
    Recommender system model
    """
    
    def __init__(
        self, 
        n_users: int = 2,
        steps: int = 1,
        priority: str | None = None,
        dummy: bool = False,
        seed: int | None = None,
        thresholds: tuple[int, int, int] = [5, 20, 50],
        ignorant_proportion: float = 1.0,
        rec_engine: str = "content-based",
        df: pd.DataFrame = pd.DataFrame(),
        df_items: pd.DataFrame = pd.DataFrame(),
        df_users: pd.DataFrame = pd.DataFrame(),
        initial_store_path: list[str] | None = None,
        n_recs: int = 50,
        social_influence: bool = False,
        run_type: str = "results",
        verbose: bool = False
    ):
        """
        Create a new recommender system model instance

        Args:
            n_users: number of users to simulate as agents
            steps: number of steps per simulation run
            priority: item priority for hidden agenda
            dummy: use of pre-loaded data for faster data loading
            seed: random state seed for sampling users
            thresholds: book limit thresholds for low-mid and mid-high reader personas
            ignorant_proportion: proportion of users that are ignorant to the intentions of algorithm
            rec_engine: type of RS ('content-based' or 'collaborative-filtering')
            df: model df if already loaded
            df_items: items df if already loaded
            df_users: users df if already loaded
            initial_store_path: path to preloaded files or None to store new files
            n_recs: number of recommendations to return 
            social_influence: whether recommendations can be prioritized based on social influence
            run_type: 'results' or 'sensitivity' for sensitivity analysis
        """
        
        # Model initialization
        if verbose:
            print("Initializing model...\n")
        super().__init__()
        self.num_users = n_users
        self.schedule = mesa.time.RandomActivation(self)
        self.steps = steps
        self.priority = priority
        self.csv_filepaths = []
        self.rec_engine = rec_engine
        self.n_recs = n_recs
        self.social_influence = social_influence
        self.run_type = run_type
        self.verbose = verbose

        # Check at least 2 users
        if self.num_users < 2:
            raise Exception("At least 2 users expected.")
        
        # Model dataframe extraction
        if df.empty:
            df = get_model_df(
                sample_users=n_users, 
                dummy=dummy, 
                seed=seed, 
                thresholds=thresholds, 
                ignorant_proportion=ignorant_proportion
            )
        
        # Items dataframe extraction
        if df_items.empty:
            df_items = get_items_df(
                df=df, 
                priority=self.priority,
                verbose=self.verbose
            )
        len_df_items = len(df_items)
        
        # Users dataframe extraction
        if df_users.empty:
            df_users = get_users_df(
                df=df, 
                df_items=df_items, 
                thresholds=thresholds, 
                n_recs=self.n_recs, 
                social_influence=self.social_influence,
                ignorant_proportion=ignorant_proportion,
                seed=seed,
                verbose=self.verbose
            )
        len_df_users = len(df_users)
        
        # User agents creation
        if self.verbose:
            print("Creating user agents...")
        df_users["unique_id"] = range(1, len_df_users + 1)
        user_agents = df_users.apply(self.create_user, axis=1)
        for a in user_agents:
            self.schedule.add(a)
        if self.verbose:
            print(f"    - Users added")
        
        # Item agents creation
        if self.verbose:
            print("Creating item agents...")
        df_items["unique_id"] = range(len_df_users + 1, len_df_users + 1 + len_df_items)
        item_agents = df_items.apply(self.create_item, axis=1)
        for i in item_agents:
            self.schedule.add(i)
        if self.verbose:
            print(f"    - Items added")
            print("Finished model initialization!")

        # Data collection setup
        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "agent_type": lambda a: get_agent_type(a),
                "vector": lambda a: get_vector(a),
                "user_books_consumed": lambda a: get_books_consumed(a),
                "item_n_read": lambda a: get_item_n_read(a),
                "item_n_reviews": lambda a: get_item_n_reviews(a),
                "item_mean_rating": lambda a: get_item_mean_rating(a)
            }
        )

        # Create results folder
        self.results = Results()
        if not initial_store_path:
            self.results.create_new_directory(run_type=run_type, verbose=self.verbose)
            self.csv_filepaths.extend(
                self.results.store(
                    prefix="initial", 
                    data=[("interactions", df), ("items", df_items), ("users", df_users)],
                    verbose=self.verbose
                )
            )
        else:
            self.results.path = initial_store_path

    def step(self) -> None:
        """
        Advance model by one step
        """
        self.schedule.step()
        self.datacollector.collect(self)

    def create_user(self, user_row: pd.Series) -> UserAgent:
        """
        Create user agent
        """
        return UserAgent(user_row, self)

    def create_item(self, item_row: pd.Series) -> ItemAgent:
        """
        Create item agent
        """
        return ItemAgent(item_row, self)

    def run_model(self) -> None:
        """
        Run model simulation
        """
        for i in range(self.steps):
            self.step()
            print(f"Step {i + 1}/{self.steps} executed.", end="\r")
        self.csv_filepaths.extend(
            self.results.store(
                prefix="run", data=[("raw", self.get_raw_df())], verbose=self.verbose
            )
        )

    def get_raw_df(self) -> pd.DataFrame:
        """
        Get raw dataframe with results
        """
        df_raw = self.datacollector.get_agent_vars_dataframe().copy()
        return df_raw
    
    def get_processed_df(self) -> pd.DataFrame:
        """
        Get processed dataframe with results
        """
        df_vectors = self.datacollector.get_agent_vars_dataframe().copy()
        df_vectors.dropna(inplace=True)
        
        # Convert vector to single dimension pandas series
        df_vectors["vector"] = df_vectors["vector"].apply(lambda x: x[0])
        df_vectors_wide = df_vectors["vector"].apply(pd.Series)
        df_vectors_wide.columns = get_categories()
        return df_vectors_wide

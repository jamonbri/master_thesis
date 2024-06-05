from __future__ import annotations
import mesa
import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from utils import unit_normalize_vector

class ItemAgent(mesa.Agent):
    """
    Item agent model
    """
    
    def __init__(
        self, 
        item_row: pd.Series,
        model: mesa.Model
    ) -> None:
        """
        Create a new item agent instance

        Args:
            item_row: pandas series row from interactions dataframe with columns: 
                unique_id: unique ID of item
                is_read: number of users that have read the book
                rating: average rating 
                is_reviewed: number of users that have reviewed the book
                priority: hidden priority of item
                vector: category vector as numpy array
        """
        super().__init__(unique_id=item_row["unique_id"], model=model)
        self.book_id = item_row.name
        self.n_read = item_row["is_read"]
        self.mean_rating = item_row["rating"]
        self.n_reviews = item_row["is_reviewed"]
        self.priority = item_row["priority"]
        self.vector = item_row["vector"]

    def normalize_vector(self) -> np.array:
        """
        Normalize category vector between 0 and 1
        """
        return (self.vector - self.vector.min()) / (self.vector.max() - self.vector.min())

    def update(self, review: float | None = None) -> None:
        """
        Update item after interaction with user
        """
        self.n_read += 1
        self.n_reviews += 1 if review else 0
        self.mean_rating += (review / self.n_reviews) if review else 0


class UserAgent(mesa.Agent):
    """
    User agent model
    """
    
    def __init__(
        self, 
        user_row: pd.DataFrame,
        model: mesa.Model
    ) -> None:
        """
        Create a new user agent instance

        Args:
            user_row: pandas series row from interactions dataframe with columns: 
                unique_id: unique ID of item
                is_read: number of books read by user
                rating: average rating given by user
                is_reviewed: number of books reviewed by user
                book_id: list of book IDs interacted by user
                vector: category vector as numpy array
                rec_proba: probability of getting a recommendation
        """
        super().__init__(unique_id=user_row["unique_id"], model=model)
        self.user_id = user_row.name
        self.books = user_row["book_id"]
        self.n_reviews = user_row["is_reviewed"]
        self.mean_rating = user_row["rating"]
        self.n_books = user_row["is_read"]
        self.vector = user_row["vector"]
        self.read_proba = user_row["read_proba"]
        self.ignorant = user_row["ignorant"]
        self.similarities = user_row["similarities"]
        self.should_update_similarities = False
        self.books_consumed = [] 
        self.following = user_row["following"] or []
    
    def get_read_probability(self) -> float:
        """
        Calculate read probability as proportion of read books from total interacted
        """
        return self.read_proba

    def get_review_probability(self) -> float:
        """
        Calculate review probability as proportion of reviewed books from total read
        """
        return self.n_reviews / self.n_books if self.n_books else 0

    def calculate_cosine_similarity(self, agent_b: UserAgent | ItemAgent) -> np.ndarray:
        """
        Calculate cosine similarity between user's own vector and other agent's vector (user or item)
        """
        return cosine_similarity(self.vector, agent_b.vector)

    def find_most_similar_agent(self) -> UserAgent | None:
        """
        Find most similar agent by comparing cosine similarity between vectors
        """
        max_similarity = -1
        most_similar_agent = None
        for other_agent in self.model.get_agents_of_type(UserAgent):
            if other_agent != self:
                similarity = self.calculate_cosine_similarity(other_agent)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_agent = other_agent
        return most_similar_agent

    def get_recommendations(self, n: int) -> dict:
        """
        Get item recommendations as dict with item ID and probability

        Args:
            n: number of recommendations
        """
        alpha = 2
        recs = {}
        social_influence_books = self.get_social_influence_books()
        top_books = self.get_top_books(n - len(social_influence_books))
        rec_list = social_influence_books[:]
        rec_list.extend(book for book in top_books if book not in rec_list)
        for idx, rec in enumerate(rec_list):
            prob = (alpha - 1) * (alpha ** (-idx - 1))
            recs.update({rec: prob})
        return recs

    def get_social_influence_books(self) -> list[int]:
        """
        Get list of books from social influencers
        """
        if not self.model.social_influence:
            return []
        rec_list = []
        for user_id in self.following:
            agent = [agent for agent in self.model.get_agents_of_type(UserAgent) if agent.user_id == user_id]
            rec_list.extend(book for book in agent[0].books_consumed if book not in rec_list)
        return rec_list

    def pick_choice(self, recs: dict) -> ItemAgent | None:
        """
        Pick choice from dictionary of books and probabilities based on random choice from probabilities

        Args:
            recs: dictionary of recommendations
        """
        recs = {k: v for k, v in recs.items() if k not in self.books and k not in self.books_consumed}
        if not recs:  # there could be a slight chance that the agent can't get more recommendations
            return
        books = list(recs.keys())
        probabilities = list(recs.values())
        choice = random.choices(books, weights=probabilities, k=1)[0]
        item = [i for i in self.model.get_agents_of_type(ItemAgent) if i.book_id == choice]
        return item[0]

    def get_top_books(self, n_books) -> list[int]:
        """
        Get the top books by vector similarity with agent or all items

        Args:
            n_books: number of books to return
            all: boolean if the books should be compared to agent (False) or all items (True)
        """
        if self.model.rec_engine == "content-based":
            if not self.should_update_similarities:
                return list(self.similarities.keys())
            return self.update_similarities(n_books)
        elif self.model.rec_engine == "collaborative-filtering":
            most_similar_agent = self.find_most_similar_agent()
            items = most_similar_agent.books
            sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
            return [item[0] for item in sorted_items[:n_books]]

    def update_similarities(self, n_books) -> list[int]:
        """
        Updates cosine similarities for user given all items and returns top n

        Args:
            n: number of items to return
        """
        items = self.model.get_agents_of_type(ItemAgent)
        items_matrix = np.array([book.vector.flatten() for book in items])
        items_matrix = np.array([unit_normalize_vector(v) for v in items_matrix])
        vector = unit_normalize_vector(self.vector.flatten())
        similarities = np.dot(items_matrix, vector)
        results = {item.book_id: round(cosine_sim, 4) for item, cosine_sim in zip(items, similarities)}
        sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
        self.similarities = dict(sorted_items[:n_books])
        self.should_update_similarities = False
        return [item[0] for item in sorted_items[:n_books]]
    
    def update(self, item: ItemAgent) -> float:
        """
        Update user agent after interaction with item

        Args:
            item: item agent interacted with
        """
        item_vector = np.where(item.vector > 0, 1, 0)
        self.vector += item_vector
        similarity = self.calculate_cosine_similarity(item)
        self.books.update({item.book_id: similarity[0][0]})
        self.books_consumed.append(item.book_id)
        self.should_update_similarities = True
        return similarity[0][0]
    
    def step(self) -> None:
        """
        Single step of user agent
        """
        
        # Should agent get recommendations?
        
        if random.random() < self.get_read_probability():
            recs = self.get_recommendations(self.model.n_recs)
            book = self.pick_choice(recs)
            if not book:
                return

            similarity = self.update(book)
            
            # Should agent review book?
            
            if random.random() < self.get_review_probability():
                review = round(similarity * 5) / 5
            else:
                review = None
            book.update(review)
    
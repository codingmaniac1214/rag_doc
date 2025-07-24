import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/Users/nimishgupta/Documents/rag_doc/models"
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
import xgboost as xgb
import pickle
import numpy as np

class RelevanceModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        logging.debug("Initialized XGBoost regressor")

    def train(self, X, y):
        """
        Train the relevance model on feature dictionaries and labels.
        Args:
            X (list): List of feature dictionaries.
            y (list): List of relevance scores.
        """
        try:
            # Convert feature dictionaries to matrix
            feature_names = X[0].keys() if X else []
            X_matrix = np.array([[x.get(f, 0) for f in feature_names] for x in X])
            y_array = np.array(y)
            self.model.fit(X_matrix, y_array)
            logging.debug(f"Trained model with {len(X)} samples, features: {feature_names}")
        except Exception as e:
            logging.error(f"Failed to train model: {e}")
            raise

    def predict(self, X):
        """
        Predict relevance scores for feature dictionaries.
        Args:
            X (list): List of feature dictionaries.
        Returns:
            np.array: Predicted relevance scores.
        """
        try:
            feature_names = X[0].keys() if X else []
            X_matrix = np.array([[x.get(f, 0) for f in feature_names] for x in X])
            scores = self.model.predict(X_matrix)
            logging.debug(f"Predicted scores for {len(X)} samples")
            return scores
        except Exception as e:
            logging.error(f"Failed to predict scores: {e}")
            raise

    def save(self, path):
        """
        Save the model to a file.
        Args:
            path (str): Path to save the model.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            logging.debug(f"Saved model to {path}")
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            raise

    def load(self, path):
        """
        Load the model from a file.
        Args:
            path (str): Path to the model file.
        """
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            logging.debug(f"Loaded model from {path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
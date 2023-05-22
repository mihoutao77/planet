import streamlit as st
import streamlit_book as stb
from pathlib import Path

# Set multipage
current_path = Path(__file__).parent.absolute()

# Supervised Learning
stb.set_book_config(menu_title="machine learning algorithm",
                    menu_icon="",
                    options=[
                            "Machine Learning",
                            "linear regression",
                            "logistic regression",
                            "decision tree",
                            "random forest",
                            "Support Vector Machines",
                            "K nearest neighbor algorithm",
                            "K-means clustering algorithm",
                            "hierarchical clustering algorithm"
                            ],
                    paths=[
                        current_path / "pages/home.py",
                        current_path / "pages/linear_regression.py",
                        current_path / "pages/logistic_regression.py",
                        current_path / "pages/decision_tree.py",
                        current_path / "pages/random_forest.py",
                        current_path / "pages/support_vector_machine.py",
                        current_path / "pages/knearest_neighbors.py",
                        current_path / "pages/kmeans_clustering.py",
                        current_path / "pages/hierarchical_clustering.py"
                          ],
                    icons=[
                          "brightness-high-fill",
                          "",
                          "",
                          "",
                          "",
                          "",
                          "",
                          "",
                          "",
                          "trophy"
                          ],
                    save_answers=False,
                    )



with st.sidebar:

    st.sidebar.title("Tips")
    st.sidebar.info(
        """
        Thanks to 小夫
        
        Thanks ML01 2023 Spring

        Learn More：https://open-academy.github.io/machine-learning/intro.html
        
        """
    )




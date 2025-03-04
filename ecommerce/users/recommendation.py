from surprise import  SVD
import joblib
model = SVD()
model = joblib.load(r'C:\Users\Dell\OneDrive\Desktop\git\college project\ecommerce\lightfm_model.pkl')
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def get_recommendations(user_id, model, all_product_ids):
    
    predictions = [model.predict(user_id, product_id) for product_id in all_product_ids]
    
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    
    top_recommendations = [prediction.iid for prediction in predictions[:9]]  
    return top_recommendations









df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\git\college project\ecommerce\ecommerce\data\user_product_interactions_no_rating.csv")
action_map = {'click': 1, 'view': 1, 'buy': 2}
df['action_value'] = df['action'].map(action_map)
interaction_matrix = df.pivot_table(index='user_id', columns='Product_id', values='action_value', aggfunc='max', fill_value=0)
product_similarity = cosine_similarity(interaction_matrix.T)
product_similarity_df = pd.DataFrame(product_similarity, index=interaction_matrix.columns, columns=interaction_matrix.columns)
product_similarity_df.to_csv("Matrix.csv")
def recommend_products(user_id, top_n=10):
    
    user_interactions = interaction_matrix.loc[user_id]
    interacted_products = user_interactions[user_interactions > 0].index.tolist()
    product_scores = {}
    for product in interaction_matrix.columns:
        if product not in interacted_products:
            
            similarity_score = 0
            for interacted_product in interacted_products:
                similarity_score += product_similarity_df.loc[product, interacted_product] * user_interactions[interacted_product]
            product_scores[product] = similarity_score

    
    sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True) 
    top_products = [product for product, score in sorted_products[:top_n]]
    return top_products


def get_matrix(user_id):
    """
    Returns the interaction matrix for a given user and the similarity matrix for each product from which
    the recommendation score was calculated.
    """ 
    user_interactions = interaction_matrix.loc[user_id]
    interacted_products = user_interactions[user_interactions > 0].index.tolist()
    product_matrices = {}
    for product in interaction_matrix.columns:
        if product not in interacted_products:
            similarity_matrix = {}
            for interacted_product in interacted_products:
                similarity_matrix[interacted_product] = {
                    "Similarity Score": product_similarity_df.loc[product, interacted_product],
                    "User Interaction Value": user_interactions[interacted_product]
                }
            product_matrices[product] = similarity_matrix

    return user_interactions, interacted_products, product_matrices
    




import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split

# Load your CSV
df = pd.read_csv("/content/user_product_interactions_no_rating.csv")


action_mapping = {'click': 1, 'view': 0.5, 'buy': 2}
df['action'] = df['action'].map(action_mapping)


reader = Reader(rating_scale=(0, 2))  

data = Dataset.load_from_df(df[['user_id', 'product_id', 'action']], reader)


trainset, testset = train_test_split(data, test_size=0.2)


model = SVD()
model.fit(trainset)
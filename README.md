collaborative filtering : similarity of same user on different product

1. Setup environments and packages

open the project directory
create a env by shift + ctrl + p --> Python: Create Environment --> Venv
select the new env by shift + ctrl + p --> Python: Select Interpreter --> Venv
type the command in the terminal to install packages: pandas, scipy, scikit-learn, ...
> pip install -r requirements.txt
    
check if there is no missing packages


2. Preprocess data for this model

save the json file to csv file, getting the values e.g., avg_rating and activeness
> python json2csv.py ./data/<json_file>


3. Start the model
> python collab_filter_nouser.py

The user-based version may require some testing cases or updated "also_buy" and "also_view" data.
Current it is not yet deployable.
> python collab_filter_main.py


#====================================#
FAQ with ChatGPT

1. Collaborative filtering is a technique commonly used in recommender systems to make personalized product recommendations. It relies on the idea that users with similar preferences in the past are likely to have similar preferences in the future. In the context of a product recommender system, collaborative filtering analyzes the patterns and similarities among users' interactions with products to make recommendations.

Collaborative filtering can be implemented in two main approaches:

- User-based collaborative filtering: It identifies users who have similar preferences and recommends products that those similar users have liked or interacted with. For example, if User A and User B have similar ratings and preferences for certain products, the system might suggest products that User B has rated highly to User A.
- Item-based collaborative filtering: It identifies products that are similar based on user interactions and recommends products that are similar to the ones a user has already rated or interacted with. For instance, if a user has rated Product X highly, the system might suggest similar products that other users have rated highly as well.


2. Inputs needed for collaborative filtering in a recommender system typically include:

- User-Product Interactions: Data on user interactions with products, such as product ratings, reviews, or purchase history.
- User-User or Item-Item Similarity Matrix: A matrix that quantifies the similarity between users or items based on their interactions. This matrix helps identify similar users or similar items for making recommendations.
- User Preferences: Optional additional information about users' preferences, demographics, or characteristics that can be used to enhance the recommendations.

3. Visualizing the calculation and results of collaborative filtering can be done in various ways. Some common approaches include:

- Heatmaps: Visualizing the similarity matrix as a heatmap, where each cell represents the similarity between two users or items. This helps identify patterns and clusters of similar users or items.
- Network Graphs: Representing user-user or item-item relationships as a network graph, where nodes represent users or items, and edges represent their similarity. This can provide a visual understanding of the recommendation connections.
- Top-N Recommendations: Displaying a list of top-N recommended products for a given user based on collaborative filtering. This allows users to see personalized recommendations.

4. If you only have product scores and ratings without any user preferences, you can still build a product-based recommender system. In this case, you can apply item-based collaborative filtering. The system analyzes the similarity between products based on their ratings and recommends products that are similar to the ones a user has interacted with or rated highly.

To implement this, you would typically calculate the similarity between products using a technique such as cosine similarity or Pearson correlation. You can then identify the most similar products to a given product and recommend those to users who have interacted with the original product.

However, without any user preferences or interactions, the recommendations will be more general and not personalized to individual users. It will be based solely on the similarities between products and not take into account specific user preferences or characteristics.
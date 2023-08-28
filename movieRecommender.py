'''
@article{10.1145/2827872,
author = {Harper, F. Maxwell and Konstan, Joseph A.},
title = {The MovieLens Datasets: History and Context},
year = {2015},
issue_date = {January 2016},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {5},
number = {4},
issn = {2160-6455},
url = {https://doi.org/10.1145/2827872},
doi = {10.1145/2827872},
journal = {ACM Trans. Interact. Intell. Syst.},
month = dec,
articleno = {19},
numpages = {19},
keywords = {Datasets, recommendations, ratings, MovieLens}
}
'''

import pandas
import numpy

movies_dataframe = pandas.read_csv("movies.csv")
ratings_dataframe = pandas.read_csv("ratings.csv")

def find_user_distance(userId1, userId2):
    user_1_ratings = find_user_ratings(userId1)
    user_2_ratings = find_user_ratings(userId2)

    ratings_comparison = user_1_ratings.join(user_2_ratings, lsuffix = "_user1", rsuffix = "_user2").dropna() #join user ratings

    user_1_vector = ratings_comparison["rating_user1"].to_numpy()
    user_2_vector = ratings_comparison["rating_user2"].to_numpy()
    absolute_user_distance = numpy.linalg.norm(user_1_vector - user_2_vector) #calculate euclidean distance between user ratings

    return [userId1, userId2, absolute_user_distance]

def find_user_ratings(userID):
    user_ratings = ratings_dataframe.query(f"userId == {userID}")
    return user_ratings[["movieId", "rating"]].set_index("movieId") #return user ratings

def find_relative_distance(userId):
    users = ratings_dataframe["userId"].unique() #find all users
    users = users[users != userId] #remove passed in user from list of users
    distances = [find_user_distance(userId, every_other_user_id) for every_other_user_id in users] #find distances between user and all other users

    return pandas.DataFrame(distances, columns=["masterUserId", "userId", "distance"]) #return dataframe of distances between our master user and all other users

def find_closest_users(userId, number_of_users):
    relative_distances = find_relative_distance(userId)
    relative_distances.sort_values("distance", inplace=True) #sort distances
    return relative_distances.head(number_of_users) #return top n closest users

def make_recommendation(userId):
    user_ratings = find_user_ratings(userId)
    closest_users = find_closest_users(userId, 10)
    most_similar_user_id = closest_users.iloc[0]["userId"] #find most similar user

    closest_user_ratings = find_user_ratings(most_similar_user_id) #find most similar user's ratings
    unwatched_movies = closest_user_ratings.drop(user_ratings.index, errors= "ignore") #find movies the most similar user has watched that the user has not watched

    unwatched_movies = unwatched_movies.sort_values("rating", ascending=False) #sort movies by rating with highest on top
    unwatched_movies = unwatched_movies.join(movies_dataframe)

    return unwatched_movies.head(10) #return top 10 movies

print(make_recommendation(9))

movies_dataframe.set_index("movieId", inplace = True)
total_rating_count = ratings_dataframe["movieId"].value_counts()

movies_dataframe["totalRatingCount"] = total_rating_count   #add total rating count column to movies dataframe

average_movie_ratings = ratings_dataframe.groupby("movieId").mean()["rating"]
movies_dataframe["averageRating"] = average_movie_ratings #add average rating column to movies dataframe

movies_dataframe.sort_values(["totalRatingCount", "averageRating"], ascending=False) #sort movies dataframe by total rating count
min_ratings_subset = movies_dataframe.query(f"totalRatingCount >= {100}") #query movies dataframe for movies with over 100 ratings
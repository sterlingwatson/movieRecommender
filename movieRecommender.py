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



movies_dataframe.set_index("movieId", inplace = True)
total_rating_count = ratings_dataframe["movieId"].value_counts()

movies_dataframe["totalRatingCount"] = total_rating_count   #add total rating count column to movies dataframe
#movies_dataframe.sort_values("totalRatingCount", ascending = False, inplace = True) #sort movies dataframe by total rating count

average_movie_ratings = ratings_dataframe.groupby("movieId").mean()["rating"]
movies_dataframe["averageRating"] = average_movie_ratings #add average rating column to movies dataframe

movies_dataframe.sort_values(["totalRatingCount", "averageRating"], ascending=False) #sort movies dataframe by total rating count
min_ratings_subset = movies_dataframe.query(f"totalRatingCount >= {100}") #query movies dataframe for movies with over 100 ratings
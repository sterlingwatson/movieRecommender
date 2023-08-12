import pandas

movies_dataframe = pandas.read_csv("movies.csv")
ratings_dataframe = pandas.read_csv("ratings.csv")

movies_dataframe.set_index("movieId", inplace = True)
total_rating_count = ratings_dataframe["movieId"].value_counts()

movies_dataframe["totalRatingCount"] = total_rating_count   #add total rating count column to movies dataframe
#movies_dataframe.sort_values("totalRatingCount", ascending = False, inplace = True) #sort movies dataframe by total rating count

average_movie_ratings = ratings_dataframe.groupby("movieId").mean()["rating"]
movies_dataframe["averageRating"] = average_movie_ratings #add average rating column to movies dataframe

movies_dataframe.sort_values(["totalRatingCount", "averageRating"], ascending=False) #sort movies dataframe by total rating count
min_ratings_subset = movies_dataframe.query("totalRatingCount >= {100}") #query movies dataframe for movies with over 100 ratings
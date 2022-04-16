#Hybrid Recommender System

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

movie = pd.read_csv('movie.csv')
rating = pd.read_csv('rating.csv')

#Verilerimizi tek bir dataframe'de toplayalım.
df = movie.merge(rating, how="left", on="movieId")

#Veri setini inceleyelim.
df.head()
df["title"].nunique() #27262 film var.

#Her bir film için kaç yorum olduğunu bir dataframe'de toplayalım.
rating_counts = pd.DataFrame(df["title"].value_counts())
rating_counts.head()

#Belirli sayıdan az yorum alan filmleri çıkararak evreni daraltalım.
rare_movies = rating_counts[rating_counts["title"] <= 1000].index #Yorum sayısı 1000 ve daha küçük filmlerin isimlerini seçtik. İkinci adımda bunları çıkaracağız.
common_movies = df[~df["title"].isin(rare_movies)]

#Kullanıcıları satırlara, filmleri sütunlara ve yorumları değerlere atayacak şekilde pivot yapalım.
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

common_movies["title"].nunique() #3159 yorum vardı.
len(user_movie_df.columns) #3159 yorum var.

#Kullanıcı ID'si görevlerin hiç birinde verilmediği için, kullanıcıyı random seçelim.
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=40).values) #her seferinde aynı örneği alabilmek için random state kullanıyoruz. Derstekinden farklı olsun diye random_state 40 alındı.
#UserId: 67599

#Kullanıcının İzlediği Filmlerin Bulunması:
random_user_df = user_movie_df[user_movie_df.index == random_user]

#Kullanıcının izlediği filmleri liste olarak almak:
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

#Kullanıcının İzlediği Film Sayısını bulalım.
len(movies_watched) #67599 id'li kullanıcı 33 film izlemiş.

#Aynı filmleri izleyen diğer kullanıcıların verisine ve Id'lerine erişiniz.

#67599 id'li kullanıcının izlediği filmleri user_movie_df'ten çekelim.
movies_watched_df = user_movie_df[movies_watched]

#Bu filmleri izleyen kullanıcıların kaç tane film izlediği bilgisini alalım.
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head()
user_movie_count["userId"].nunique()

#Bu kullanıcılardan 67599 id'li kullanıcının izlediği filmlerin %60'ından fazlasını izleyenleri bulalım.
perc_user = len(movies_watched)*0.6
users_same_movies = user_movie_count[user_movie_count["movie_count"]>perc_user]["userId"]
users_same_movies.head()
users_same_movies.count() #2540 kullanıcıya eriştik.

#Öneri yapılacak kullanıcı ile en benzer kullanıcıları belirleyiniz.
#İlk olarak kullanıcımız ve diğer kullanıcıların verilerini bir araya getirelim.

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)], random_user_df[movies_watched]])

#Kullanıcıların puan verme davranışına göre korelasyon df'ini oluşturalım.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])

#Index isimlerini değiştirip, indexi sıfırlayalım.
corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()

#Random seçtiğimiz kullanıcımız 67599 ile en benzer beğenme davranışını gösteren kullanıcıları bulup sıralayalım.
#En benzer beğenme davranışı için korelasyonu 0.65 üstü kullanıcıları seçiyoruz.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values("corr", ascending=False)
top_users.rename(columns = {"user_id_2":"userId"}, inplace = True)
top_users.count() #Korelasyonu .65 üstü 26 kullanıcı bulduk. Fakat kendi kullanıcımız bu sayıya dahil.

#Rating df'si ile top_user'ı birleştirelim.

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")

#Random çektiğimiz kullanıcıyı çıkaralım.

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].nunique() #Kendi kullanıcımızı çıkarınca 21 farklı kullanıcı kaldı.

#Weighted Average Recommendation Score'u hesaplayınız ve ilk 5 filmi tutunuz.
#Weighted Average Recommendation Score: Korelasyon ve rating değerlerinin etkisini bir arada tutan skordur.

top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]
top_users_ratings.head()

#Oluşturulan weighted_rating'i filmler bazında ortalamasını alıp kayıt edelim.

recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating":"mean"})
recommendation_df.reset_index(inplace=True)

#weighted_rating'i 3.5 üstü filmleri sıralayalım.

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

#Filmlerin isimlerini ekleyelim.

movies_to_be_recommend = movies_to_be_recommend.merge(movie[["movieId","title"]])

user_based_output = movies_to_be_recommend[["movieId","title"]].head()

#Item Based Filtering ile 5 Öneri Yapılması

#Item Based Filteringte kullanılacak filmi seçilmesi.
movie_id = rating[(rating["userId"] ==random_user) & (rating["rating"]==5.0)].sort_values("timestamp", ascending=False)["movieId"][0:1].values[0]

#Bu filmin adını bulalım.
movie_name = common_movies[common_movies["movieId"] == movie_id].sort_values("rating", ascending=False)["title"][0:1].values[0] #'Pulp Fiction (1994)'

#Bu filme yapılan yorumları alalım.
movie_name = user_movie_df[movie_name]

#Bu film ile korelasyonu en yüksek 5 filmi alalım. Not: 1. si kendisi olacağı için head 6 alınacak

movies_from_item_based = user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(6)

movies_from_item_based[1:6].index
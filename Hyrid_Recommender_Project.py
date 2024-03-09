
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recommender yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.


#############################################
# Görev 1: Verinin Hazırlanması
#############################################
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("expand_frame_repr", False)


# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
movie = pd.read_csv(r'C:\Users\muhammet.guneri\Desktop\Modül_4_Tavsiye_Sistemleri\Hybrid Recommender System\datasets\movie.csv')
movie.head()
print(movie.shape) #27278 tane film

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
rating = pd.read_csv(r'C:\Users\muhammet.guneri\Desktop\Modül_4_Tavsiye_Sistemleri\Hybrid Recommender System\datasets\rating.csv')
rating.head()
print(rating.shape) #20 Milyon film değerlendirmesi

# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.

df = movie.merge(rating, how="left", on="movieId") #movie veri setiyle rating veri setini movieId'lerine birleştirip, yeni bir dataframe oluşturdum.
# tam tersini yaptığımızda sorun oluyor. Neden?
df.head()
print(df.shape)


# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında
# olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.
comment_counts = pd.DataFrame(df["title"].value_counts())
print(comment_counts)



# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz
rare_movies = comment_counts[comment_counts["count"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.head()
print(common_movies.shape)

# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["movieId"], values="rating")
user_movie_df.head()


# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım
def create_user_movie_df ():
    movie = pd.read_csv(r'C:\Users\muhammet.guneri\Desktop\Modül_4_Tavsiye_Sistemleri\Hybrid Recommender System\datasets\movie.csv')
    rating = pd.read_csv(r'C:\Users\muhammet.guneri\Desktop\Modül_4_Tavsiye_Sistemleri\Hybrid Recommender System\datasets\rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["movieId"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df() #yukarıdaki işlemleri tek seferde fonksiyonla yapıyorum.

#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values) # user_Id: 28941 rastgele olarak seçiyorum.


# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
print(random_user_df.shape)

# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.
movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list() # 28941 nolu izlenen filmlerin movieId'leri


#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
print(movies_watched_df.shape) #seçtiğim kullanıcıyla aynı filmleri izlemiş 138493 kullanıcı, seçtiğim kullanıcının izlediği 33 film

# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.


user_movie_count = movies_watched_df.T.notnull().sum() #her bir kullanıcının kaç film izlediği bilgisini veriyor.
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head()

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.

perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
len(users_same_movies) #seçilen kullanıcının izlediği filmlerin %60'ını veya daha fazlasını
# izleyen benzer kullanıcıların sayısı: 4139


#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren
# kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
print(final_df.shape) #4139 kullanıcı ve 33 film


# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.

corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()

corr_df.head()

#corr_df[corr_df["user_id_1"] == random_user]


# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları
# filtreleyerek top_users adında yeni bir dataframe oluşturunuz.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by="corr", descending=False) #korelasyona göre büyükten küçüğe sıralama
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.head()
print(top_users.shape) #65 tane kullanıcı seçtiğim kullanıcı ile istediğim koşulu sağlıyor.


# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz
rating.head() #hatırlamak için
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user] #seçili olan kullanıcımı çıkarıyorum
top_users_ratings.nunique() #64 kullanıcı, 5990 filme, 10 farklı değerlendirme vermişler
top_users_ratings.head()


#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating
# adında yeni bir değişken oluşturunuz.

top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]
top_users_ratings.head()


# Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin
# ortalama değerini içeren recommendation_df adında yeni bir dataframe oluşturunuz.

recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()


# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz
# ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5] \
    .sort_values(by="weighted_rating", ascending=False)
movies_to_be_recommend.head(5)


# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.

movie.head() #hatırlamak için
movies_to_be_recommend.merge(movie[["movieId","title"]]).head(5) #ilk 5 filmin adı

#                                    title
#1                             Lamerica (1994)
#2                              Whatever (1998)
#3              Incredible Journey, The (1963)
#4                         She's All That (1999)
#5                          Tumbleweeds (1999)


#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170

# Adım 1: movie,rating veri setlerini okutunuz.
movie = pd.read_csv(r'C:\Users\muhammet.guneri\Desktop\Modül_4_Tavsiye_Sistemleri\Hybrid Recommender System\datasets\movie.csv')
movie.head()

rating = pd.read_csv(r'C:\Users\muhammet.guneri\Desktop\Modül_4_Tavsiye_Sistemleri\Hybrid Recommender System\datasets\rating.csv')
rating.head()


# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5)].sort_values(by="timestamp", ascending=False)\
               ["movieId"][0:1].values[0] #7044 nolu film

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
movie[movie["movieId"] == movie_id]["title"].values[0] #7044 nolu film Wild ad Heart (1990) adlı film imiş.
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]]["title"].values[0]
movie_df

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.

user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

item_based_recommender(movie_id, user_movie_df)
# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.


movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)

# 1'den 6'ya kadar. 0'da filmin kendisi var. Onu dışarda bıraktık.
movies_from_item_based[1:6].index

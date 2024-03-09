
#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih




#########################
# GÖREV 1: Veriyi Hazırlama
#########################
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", None)
from mlxtend.frequent_patterns import apriori, association_rules


# Adım 1: armut_data.csv dosyasınız okutunuz.

df_ = pd.read_csv(r"C:\Users\muhammet.guneri\Desktop\Armut ARL_Proje\armut_data.csv")
df = df_.copy()
df.head()

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.
print(df["ServiceId"].dtype) #integer
print(df["CategoryId"].dtype) #integer
df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
#integer olan id'leri string yapıp yeni bir değişken oluşturuyorum.
df.head()


# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
# Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.

print(df["CreateDate"].dtype) #object olan tarih değişkenini tarih verisine çevirmeliyim.
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
print(df["CreateDate"].dtype) # tarih veri tipine çevirdim

df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m") # Yıl ve ay bilgisini çıkararak yeni bir değişken oluşturma
df.head()

df["SepetID"] = df["UserId"].astype(str) + "_" + df["New_Date"].astype(str)# UserId ile New_Date değişkenlerini kullanarak SepetID adlı yeni bir değişken oluşturuyorum
df.head()



#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#########################
print(df.describe().T) # veriyi inceliyoruz burada sıkıntı yok.
df.isnull().sum() #boş değer olup olmadığına bakıyorum, boş değer yok.
print(df.shape) # 162523 tane userım hizmet almış
df["UserId"].nunique() # 24826 tane müşterim 162 bin hizmeti almış.



# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

pivot = df.pivot_table(index="SepetID", columns="Hizmet", aggfunc="size", fill_value=0).head()
pivot.head()
#Bu kod, Service_Category_IDs değerlerini sütunlara dönüştürür ve SepetID değerlerini satırlara yerleştirir.
# aggfunc='size' parametresi, her bir kombinasyonun kaç kez göründüğünü sayar. Eğer bir Service_Category_IDs değeri
# belirli bir SepetID için mevcut değilse, fill_value=0 ile bu hücrelere 0 değeri atanır.
#Sonuç, her bir SepetID için, ilgili Service_Category_IDs değerlerinin varlığını veya yokluğunu
# gösteren geniş bir tablo olacaktır

#### alternatif
#df_pivot = df.groupby(["SepetID", "Hizmet"]).apply(lambda x: 1 if x["ServiceId"].any() else 0).unstack().fillna(0)
#new_df = pd.pivot_table(df, index="Sepet_ID", columns="Hizmet", values="CategoryId", aggfunc="count"). \
    #fillna(0). \
    #applymap(lambda x: 1 if x > 0 else 0)
#bu daha iyi
# buradaki x grubun dataframe'idir. ServiceId değerinin olup olmadığı kontrol eder.
df_pivot.head()

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..




# Adım 2: Birliktelik kurallarını oluşturunuz.
frequent_itemsets = apriori(df_pivot, min_support=0.01, use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

new_rules = rules[(rules["support"] > 0.01) & (rules["confidence"] > 0.1) & (rules["lift"] > 2)]



#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(new_rules, last_service):
    recommendation_list = []
    for i, product in enumerate(new_rules["antecedents"]):
        for j in list(product):
            if j == last_service:
                recommendation_list.append(list(new_rules.iloc[i]["consequents"])[0])
    return recommendation_list

last_service = "2_0"  # Kullanıcının en son aldığı hizmet
recommendation_list = arl_recommender(new_rules, last_service)
print("Önerilen hizmetler:", recommendation_list)



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
# Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
from mlxtend.frequent_patterns import apriori, association_rules

#########################
# GÖREV 1: Veriyi Hazırlama
#########################

# Adım 1: armut_data.csv dosyasınız okutunuz.
df_ = pd.read_csv("datasets/armut_data.csv")
df = df_.copy()
df.shape
df.head()
df.info()

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.
df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()

# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
# Örneğin 25446 id'li müşteri 2017'in 8.ayında aldığı 4_5, 48_5, 6_7, 47_7  hizmetler bir sepeti; 2017'in 9.ayında
# aldığı 17_5, 14_7  hizmetler başka bir sepeti ifade etmektedir.
# Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini kullanıcı bazında "_" ile birleştirirek ID adında yeni bir değişkene atayınız.
df.info()
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
# strftime() fonksiyonu, bize tarih ve zaman bilgilerini ihtiyaçlarımız doğrultusunda biçimlendirme imkanı sunar.
df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")
df["SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]

# df["UserId"].astype("str") + "_" + df["NEW_DATE"].astype("str")
df.head()

#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#########################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

# Bir kişi aynı ay içerisinde birden fazla hizmet almış olabilir. Bu sebepten ötürü çoklamaktadır.
df["SepetID"].value_counts()

invoice_product_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
# invoice_product_df = pd.pivot_table(df, index="SepetID", columns="Hizmet", aggfunc="count", fill_value=0).applymap(lambda x: 1 if x > 0 else 0)

# Adım 2: Birliktelik kurallarını oluşturunuz.
frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

# Adım 3: arl_recommender fonksiyonunu kullanarak son 1 ay içerisinde 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

# def arl_recommender(rules_df, product_id, rec_count=1):
#     sorted_rules = rules_df.sort_values(["lift", "confidence"], ascending=False)
#     recommendation_list = []
#     for i, product in enumerate(sorted_rules["antecedents"]):
#         for j in list(product):
#             if j == product_id:
#                 recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
#
#     return recommendation_list[0:rec_count]

# derste kullandığımız fonksiyon yukarıdakiydi fakat buranın bir dezavantajı var oda;

# Örneğin A, B, C ürününü alanın D, E ürününü aldığını düşünelim. A, B ürününü alanın da D, F ürününü aldığını
# düşünelim. Buradaki kod parçasında her iki kişi için düşünürsek A ürününü girerse sonuç olarak D, D dönüyordu.
# sebebi recommendation_list.append(list(sorted_rules.iloc[i]["consequents"][0])) buradaki consequents içerisinden 0.
# elemanın seçili olmasıydı. E ve F'yi görmezden geliyordu. Peki biz naptık: Öncelikle bu 0.eleman seçme işlemini
# buradan kaldırdık ve kişi A ürününü girerse sonuç olarak [D, E] ve [D, F] dönüyordu. Bunları da tek bir listede
# birleştirmek istediğimde D, E, D, F oluyordu. Orada da bir çoklama söz konusu olduğundan bu dönen listeyi küme
# içerisine aldık ve sonra tekrar listeye çevirdiğimizde tekilleştirmiş olduk.
# Sonuç olarak A ürününü giren bir kişi D, E, F ürünlerini de alabilir şeklinde bir tahmin dönüyor artık.

my_set = {1, 2, 3, 4, 4, 4}
print(my_set)

liste = ["vbo", "vbo", "miuul", "miuul"]
print({i for i in liste})

for i, product in enumerate(rules["antecedents"]):
    print(i)
    print(product)
    for j in list(product):
        print(j)

for i, product in rules["antecedents"].items():
    print(i)
    print(product)
    for j in list(product):
        print(j)


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]


arl_recommender(rules, "2_0", 2)

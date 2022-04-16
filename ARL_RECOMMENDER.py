#Association Rule Learning ile Recommender Sistem Oluşturma

import pandas as pd
# !pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

df_ = pd.read_excel("online_retail_II.xlsx",sheet_name="Year 2010-2011")
df = df_.copy()

##Verimizdeki eşik değerlerini belirleyerek aykırı gözlemlerin yerine belirlenen alt ve üst limitlerin yerleştirecek fonksiyonu yazalım.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5*interquantile_range
    low_limit = quartile3 - 1.5*interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] < up_limit), variable] = up_limit

##Veri içindeki bazı değerlerimizi kaldıracak ve veriyi işlemeye hazır hale getirecek fonksyionu yazalım.

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

#Oluşturan fonksyionlarımı uygulayalım.

df = retail_data_prep(df)


#dataframe'mizi sadece Almanya'daki işlemleri sınırlayacak şekilde işleyelim.

df_ge = df[df["Country"] == "Germany"]

##Invoice-Product Matrisimizi oluşturalım.

ge_inv_pro_df = df_ge.groupby(['Invoice', 'StockCode'])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

#Verimiz istediğimiz yapıya geldi mi, kontrol edelim.

ge_inv_pro_df.iloc[0:5,0:5]

##Sütunları ürün isimlerinden değil StockCode'lardan oluşturduğumuz için, merak ettiğimizde ürün numarasını girerek ürün ismine
#ulaşabileceğimiz fonksiyonumuz oluşturalım.

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df, 21987) #Ürün ismi: ['PACK OF 6 SKULL PAPER CUPS']

check_id(df, 23235) #Ürün ismi:['STORAGE TIN VINTAGE LEAF']

check_id(df, 22747) #Ürün ismi: ["POPPY'S PLAYHOUSE BATHROOM"]


##Apriori ile Support değerlerimizi bulalım.

frequent_itemsets = apriori(ge_inv_pro_df, min_support=0.01, use_colnames=True)

##Support değerlerimize göre azalan olarak sıralayarak değerlerimizi gözlemleyelim.
frequent_itemsets.sort_values("support", ascending=False).head()

#Association Rules ile confidence ve lift değerlerimizi hesaplayalım.

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

##Nihai çıktımızı görüntüleyelim.

rules.sort_values("support", ascending=False).head()

##Sepetteki kullanıcılar için ürün önerisi yapalım.

sorted_rules = rules.sort_values("lift", ascending=False)


recommendation_list = []


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items(): #bu döngü ile verilen id antecedents'te aranır, tam karşılığı olan ürünler consequents'ten  çağırılır.
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list}) #bu liste ile tekrar edenler kaldırılır.

    return recommendation_list[:rec_count]

#Kullanıcı 1 ürün id'si: 21987, ürün: ['PACK OF 6 SKULL PAPER CUPS']
arl_recommender(rules, 21987, 1) #['84997D']
check_id(df, "84997D") #['PINK 3 PIECE POLKADOT CUTLERY SET']
#Yorum: 1. müşterinin verdiği ürün 6 lı kafatası desenli karton bardak, öneri 3 parça çatal takımı. Gayet mantıklı bir eşleşme.


#Kullanıcı 2 ürün id'si: 23235, ürün: ['STORAGE TIN VINTAGE LEAF']
arl_recommender(rules, 23235, 1) #[22029]
check_id(df, 22029) #['SPACEBOY BIRTHDAY CARD']
#Yorum: 2. müşterinin ürünü yaprak desenli saklama kabı, öneri doğum günü kartı. Pek alakalı değil. 2. öneri var mı bakalım:
arl_recommender(rules, 23235, 2) #[22029, 20750]
check_id(df, 20750) #['RED RETROSPOT MINI CASES'] Bu öneri 1. öneriye göre çok daha ilişkili.


#Kullanıcı 3 ürün id'si: 22747, ürün: ["POPPY'S PLAYHOUSE BATHROOM"]
arl_recommender(rules, 22747, 1) #[20750]
check_id(df, 20750) #['RED RETROSPOT MINI CASES']
#Yorum: Bu müşteri içinde diğer önerilere bakalım.
arl_recommender(rules, 22747, 3) #[20750, 22423, 22555]
check_id(df, 22555) #['PLASTERS IN TIN STRONGMAN'] İlk 2 öneri çok ilişkili değil, 3. öneri daha ilişkili.
# Armut-Association-Rule-Learning


--------------------------
### İş Problemi
--------------------------

 Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
 Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
 ulaşılmasını sağlamaktadır.
 Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
 Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.

--------------------------
### Veri Seti
--------------------------


 Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
 Alınan her hizmetin tarih ve saat bilgisini içermektedir.

 UserId: ---Müşteri numarası
 
 ServiceId: ---Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
             Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
             (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
             
 CategoryId: ---Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
 
 CreateDate: ---Hizmetin satın alındığı tarih

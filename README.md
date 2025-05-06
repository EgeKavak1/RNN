## Giriş (Introduction)

Duygu analizi (sentiment analysis), metin verisindeki duygu tonunu (pozitif, negatif, nötr) belirlemeyi amaçlayan doğal dil işleme (NLP) alanının temel problemlerinden biridir. Kısa metinlerdeki duygu analizi, sosyal medya yorumları, ürün değerlendirmeleri veya basit ifadelerin otomatik olarak sınıflandırılması gibi çeşitli uygulamalara sahiptir. Tekrarlayan Sinir Ağları (RNN) ve özellikle uzun-kısa süreli bellek ağları (LSTM), metinlerin dizisel yapısını modellemek için yaygın olarak kullanılan derin öğrenme mimarileridir.

Bu çalışmada, sağlanan küçük ve özel olarak hazırlanmış ikili (pozitif/negatif) duygu veri seti üzerinde iki farklı yaklaşımı karşılaştırıyoruz:

1.  **Model 1:** NumPy kütüphanesi kullanılarak katmanları ve geri yayılım algoritması manuel olarak yazılan bir LSTM modeli.
2.  **Model 2:** PyTorch kütüphanesinin sağladığı hazır katmanlar (`nn.Embedding`, `nn.RNN`, `nn.Linear`) kullanılarak oluşturulan bir SimpleRNN modeli.

Çalışmanın amacı, her iki modelin performansını (doğruluk ve kayıp) değerlendirmek, eğitim süreçlerini karşılaştırmak ve bu özel veri seti bağlamında manuel implementasyon ile hazır kütüphane kullanımının avantaj ve dezavantajlarını tartışmaktır.

## Yöntemler (Methods)

### Veri Seti

Çalışmada kullanılan veri seti, kısa İngilizce cümleler ve bunların ikili duygu etiketlerinden (Pozitif için True, Negatif için False) oluşmaktadır. Toplam **57 eğitim** ve **20 test** örneği bulunmaktadır. Veri seti nispeten küçüktür ve belirli kelime kalıplarının (örneğin, "good", "bad", "not", "very") duygu üzerinde doğrudan etkili olduğu görülmektedir.

### Veri Ön İşleme

Modellerin metin verisini işleyebilmesi için sayısal formata dönüştürülmesi gerekmektedir. Her iki model için temel ön işleme adımları benzerdir:

1.  **Tokenizasyon:** Cümleler kelimelere (tokenlara) ayrılır. Metinler küçük harfe çevrilir ve noktalama işaretleri temizlenir.
2.  **Kelime Haznesi (Vocabulary):** Eğitim veri setindeki tüm benzersiz kelimeler toplanarak bir kelime haznesi oluşturulur. Bu hazneye ek olarak, dolgulama için `<PAD>` ve eğitim setinde görünmeyen kelimeler için `<UNK>` (bilinmeyen) özel tokenları eklenir. Her kelimeye benzersiz bir indeks atanır.
3.  **Sayısal Gösterim (Numericalization):** Her cümle, içerdiği kelimelerin indeks dizisine çevrilir.
4.  **Padding/Truncation:** RNN modelleri genellikle sabit uzunlukta girdiler bekler. Cümle indeks dizileri, veri setindeki en uzun cümle uzunluğuna (`max_sequence_length`) pad edilir (sonuna `<PAD>` indeksleri eklenir). Daha uzun cümleler baştan kesilebilir (bu veri setinde gerekmemiştir).

-   **Model 1 (Manuel LSTM) için Ön İşleme:** Manuel LSTM modeli, `forward` metoduna doğrudan kelime vektörlerinin bir dizisini bekler (manuel `backprop` metodu bu yapıyı varsayar). Bu nedenle, indeks dizileri oluşturulduktan sonra, her indeksin bir tek-sıcak (one-hot) vektöre veya ayrı bir manuel gömme katmanı tarafından üretilen bir vektöre çevrilmesi gerekir. Sağlanan `backprop` kodu, girdi olarak `(input_size, 1)` şeklinde vektörlerin bir listesini (`inputs`) bekler, bu da tipik olarak tek-sıcak veya düşük boyutlu manuel gömme çıktısıdır. Model 1'in ön işlemesi, bu vektör dizisini oluşturmayı içerir.
-   **Model 2 (PyTorch RNN) için Ön İşleme:** PyTorch modeli, `torch.nn.Embedding` katmanını kullanır. Giriş olarak padding uygulanmış indeks dizilerini alır. Gömme katmanı, bu indeksleri öğrenilebilir yoğun vektörlere çevirir. Kelime haznesi oluşturma ve padding işlemleri PyTorch DataLoader kullanılırken manuel veya yardımcı fonksiyonlarla yapılır (bu raporda sağlanan PyTorch kodunda manuel fonksiyonlar kullanılmıştır).

### Model Mimarileri

#### Model 1: LSTM (NumPy - Manuel)

NumPy kullanılarak temel LSTM hücre mimarisinin ve çıkış katmanının manuel implementasyonudur:

* **LSTM Hücresi:** Standart dört kapılı (giriş, unutma, çıkış) ve hücre durumu mekanizmasını içerir. Her adımda mevcut girdi ve önceki gizli/hücre durumunu kullanarak kapı değerlerini ve yeni hücre/gizli durumu hesaplar. Sigmoid ve Tanh aktivasyonları kullanılır.
* **Çıkış Katmanı:** Son zaman adımındaki gizli durumu alarak tek bir çıktı değeri üretir (`W_y`, `b_y` ağırlıkları).
* **İleri Yayılım (`forward`):** Girdi dizisini adım adım işleyerek kapı değerlerini, hücre durumlarını ve gizli durumları hesaplar ve bir sonraki adıma aktarır. Ara değerleri geri yayılım için saklar. Dizinin sonunda çıkış katmanı uygulanır.
* **Geri Yayılım (`backprop`):** Zaman Boyunca Geri Yayılım (BPTT) algoritmasını manuel olarak implement eder. Çıkış hatasını alır ve zincir kuralını kullanarak tüm ağırlıklar ve yanlılıklar için gradyanları hesaplar. Patlayan gradyanları önlemek için gradyan kırpma (gradient clipping) içerir. Hesaplanan gradyanlara göre SGD kullanarak ağırlıkları günceller.

#### Model 2: SentimentRNN (PyTorch)

PyTorch'un modüler yapısı kullanılarak oluşturulmuş sıralı (sequential) bir modeldir:

* **Embedding Katmanı (`nn.Embedding`):** Kelime haznesi boyutunda girdi alır ve kelime başına belirli boyutta yoğun vektör çıktısı verir (`embedding_dim`).
* **SimpleRNN Katmanı (`nn.RNN`):** Gömülü kelime vektörleri dizisini zaman adımında işler. Belirli sayıda gizli birimi (`hidden_size`) vardır. Sadece dizinin sonundaki gizli durumu döndürecek şekilde yapılandırılmıştır (`batch_first=True` ve varsayılan `return_sequences=False`).
* **Tam Bağlı Katman (`nn.Linear`):** RNN'in son gizli durum çıktısını alarak ikili sınıflandırma için 1 boyutlu ham çıktı (logit) üretir.
* **Aktivasyon:** Çıkış katmanından gelen logit'lere eğitim sırasında kayıp fonksiyonu içinde `sigmoid` uygulanır. Tahmin sırasında ise manuel olarak `sigmoid` uygulanıp 0.5 eşiği kullanılır.

### Eğitim ve Optimizasyon

* **Model 1 (Manuel LSTM):**
    * Kayıp Fonksiyonu: İkili Çapraz Entropi (Binary Cross-Entropy - BCE). Manuel `backprop` metoduna geçirilen `d_L_d_y` değeri, BCE kaybının çıkış katmanı pre-aktivasyonuna göre türevidir.
    * Optimizer: Stokastik Gradyan Azaltma (SGD). Ağırlık güncellemeleri `backprop` metodunun içinde manuel olarak yapılır (öğrenme oranı `learn_rate`, sağlanan çıktıda 0.01 olarak belirtilmiş).
    * Eğitim: Belirtilen sayıda epok (sağlanan çıktıda 1000 epoka kadar sonuçlar var), eğitim verisindeki her örnek için (veya küçük batch'ler için) manuel olarak ileri ve geri yayılım (`forward`, `backprop`) çağrılarak yapılır.
* **Model 2 (PyTorch RNN):**
    * Kayıp Fonksiyonu: `nn.BCEWithLogitsLoss` (İkili Çapraz Entropi ve Sigmoid'i birleştirir).
    * Optimizer: `optim.Adam` (öğrenme oranı 0.005).
    * Eğitim: Belirtilen sayıda epok (bu raporda 500 epokluk çalışmaya ait sonuçlar kullanılacaktır), belirlenen batch boyutu (8) ile DataLoader kullanılarak yapılır. Eğitim döngüsü PyTorch'un otomatik farklılaştırma (`.backward()`) ve optimizer (`.step()`) özelliklerini kullanır.

### Değerlendirme

Modellerin performansı, eğitim ve test veri setleri üzerindeki **doğruluk (Accuracy)** ve **kayıp (Loss - Binary Cross-Entropy)** değerleri ile ölçülmüştür. Doğruluk, doğru sınıflandırılan örneklerin toplam örnek sayısına oranıdır. Kayıp, modelin tahminlerinin gerçek etiketlerden ne kadar saptığını gösterir (daha düşük kayıp daha iyi performans demektir).

## Sonuçlar (Results)

İki modelin eğitim ve test setleri üzerindeki nihai performans sonuçları (veya sağlanan çıktılardaki son epok sonuçları) aşağıdaki tabloda gösterilmiştir:

| Model                   | Eğitim Kaybı | Eğitim Doğruluğu | Test Kaybı | Test Doğruluğu |
| :---------------------- | :----------- | :--------------- | :--------- | :------------- |
| **Model 1 (Manuel LSTM)**| **0.0010** | **1.0000** | 0.3780     | **0.9000** |
| **Model 2 (PyTorch RNN)**| ~0.3350      | ~0.8877          | ~0.3776    | **0.9000** |

*Not: Model 1 (Manuel LSTM) sonuçları sağlanan 1000 epok çıktısındaki son değerlerdir, Model 2 (PyTorch) sonuçları ise 500 epok sonundadır.*

**Model 1 (Manuel LSTM) - Epoklara Göre Performans Eğilimleri:**

Sağlanan çıktılara göre, Model 1'in (Manuel LSTM) eğitim süreci boyunca performans eğilimleri şöyledir:

| Epoch | Eğitim Kaybı | Eğitim Doğruluğu | Test Kaybı | Test Doğruluğu |
| :---- | :----------- | :--------------- | :--------- | :------------- |
| 200   | 0.669        | 0.621            | 0.722      | 0.650          |
| 300   | 0.643        | 0.690            | 0.961      | 0.550          |
| 400   | 0.407        | 0.862            | 0.577      | 0.650          |
| 500   | 0.328        | 0.828            | 0.705      | 0.650          |
| 600   | 0.130        | 0.948            | 0.737      | 0.700          |
| 700   | 0.016        | 1.000            | 0.340      | 0.900          |
| 800   | 0.003        | 1.000            | 0.331      | 0.900          |
| 900   | 0.002        | 1.000            | 0.355      | 0.900          |
| 1000  | 0.001        | 1.000            | 0.378      | 0.900          |

Model 2'nin (PyTorch RNN) epok bazlı çıktısı tam olarak sağlanmamış olsa da, genellikle hazır kütüphanelerle eğitim daha stabil ve hızlı bir yakınsama gösterir. Model 1'in çıktılarında test doğruluğunun ve kaybının eğitim sürecinin başlarında önemli dalgalanmalar gösterdiği (0.55'e düşüp tekrar 0.90'a çıkması gibi) görülmektedir. Model 1, eğitim setinde %100 doğruluğa ulaşarak tam bir uyum sağlarken (aşırı uyma potansiyeli), test doğruluğu %90'da kalmıştır.

## Tartışma (Discussion)

Her iki model de sağlanan küçük ve belirgin kalıplara sahip veri setinde oldukça başarılı olmuştur. Nihai test doğruluğu açısından her iki model de %90'a ulaşmıştır. Ancak, eğitim süreçleri ve implementasyon yaklaşımları arasında belirgin farklar bulunmaktadır.

* **Performans ve Yakınsama:** Nihai test doğruluğu aynı olsa da, Model 1'in (Manuel LSTM) eğitim sürecindeki test performansı daha fazla dalgalanma sergilemiştir. Bu durum, manuel BPTT implementasyonunun sayısal kararsızlıklarına, öğrenme oranına veya küçük batch boyutuna bağlı olabilir. PyTorch gibi kütüphanelerdeki optimize edilmiş algoritmalar ve varsayılan ayarlar genellikle daha stabil yakınsama sağlar. Model 1'in eğitim doğruluğunun %100'e ulaşması, eğitim verisine tamamen uyduğunu gösterir ki bu, daha büyük veri setlerinde aşırı uyma (overfitting) riskinin bir işaretidir. Test doğruluğunun %90'da kalması, bu aşırı uyumun test setine tam olarak genellenmediğini ancak modelin eğitim verisini ezberleme eğiliminde olduğunu gösterir. Model 2 de yüksek eğitim doğruluğuna ulaşmış olabilir, ancak sağlanan çıktılarda %100'e ulaştığı bilgisi yok.
* **Model Mimarisi (LSTM vs. SimpleRNN):** Teorik olarak LSTM'ler, SimpleRNN'lerden daha karmaşık kalıpları ve uzun süreli bağımlılıkları yakalamakta daha iyidir. Ancak, bu veri setindeki cümleler çok kısa olduğu için LSTM'in bu avantajı belirgin şekilde ortaya çıkmamıştır. Basit bir SimpleRNN bile bu veri setindeki temel duygu kalıplarını (kelime varlığı, "not" kelimesinin etkisi) etkili bir şekilde öğrenebilmiştir. Her iki mimari de bu görev için yeterli olmuştur.
* **Implementasyon Yaklaşımı:**
    * **Model 1 (Manuel LSTM):** NumPy ile bir LSTM modelini manuel olarak implement etmek ve BPTT'yi yazmak, RNN/LSTM'in iç işleyişini, kapıların ve hücre durumunun nasıl güncellendiğini, gradyanların zaman içinde nasıl yayıldığını derinlemesine anlamayı sağlar. Bu çok değerli bir öğrenme deneyimidir. Ancak, bu yaklaşım hata yapmaya daha açıktır (özellikle gradyan hesaplamalarında), sayısal kararlılığı sağlamak zordur (manuel gradyan kırpma veya diğer teknikler gerekebilir) ve genellikle optimize edilmiş kütüphane implementasyonları kadar hızlı değildir. Model 1'de BPTT'nin başarılı bir şekilde implement edilmiş olması takdire şayandır.
    * **Model 2 (PyTorch):** Hazır kütüphane katmanları kullanmak, geliştirme hızını artırır, karmaşık matematiksel hesaplamaları (otomatik farklılaştırma, optimize edilmiş matris işlemleri) kütüphaneye devreder ve genellikle daha stabil ve performanslı bir eğitim süreci sunar. Pratik uygulamalar ve büyük ölçekli problemler için standart yaklaşımdır.

Bu özel, küçük ve sentetik veri seti için her iki model de nihayetinde aynı test doğruluğuna ulaşmıştır. Bu durum, veri setinin karmaşıklığının iki modelin mimari farkını (SimpleRNN vs LSTM) belirginleştirmeye yetmediğini göstermektedir. Bunun yerine, manuel implementasyonun (Model 1) eğitim kararlılığı üzerindeki potansiyel etkisi (gözlemlenen dalgalanmalar) ve eğitim verisine tam uyum sağlama eğilimi daha dikkat çekicidir. Model 2'nin hazır kütüphane kullanımından gelen eğitim kararlılığı bu küçük veri setinde bile belirgindir.

**Sınırlılıklar ve Gelecek Çalışmalar:**

* Veri setinin çok küçük olması, modellerin genelleme yeteneğini sınırlayabilir ve aşırı uyma riskini artırır. Daha büyük ve çeşitli veri setleri üzerinde test yapmak genelleme performansını daha iyi gösterir.
* Manuel BPTT implementasyonu (Model 1), daha büyük ve karmaşık modellerde veya veri setlerinde yönetilmesi çok daha zor hale gelebilir.
* Hiperparametre optimizasyonu (gizli boyutlar, öğrenme oranı, batch boyutu, epok sayısı, gradyan kırpma değeri) her iki modelin performansı üzerinde etkili olabilir ve daha sistematik olarak yapılabilir.
* Model 2'de veri ön işleme (tek-sıcak veya manuel gömme) ile Model 1'deki `nn.Embedding` katmanının karşılaştırılması da ilgi çekici olabilir.
* Daha karmaşık mimariler (GRU, çift yönlü RNN'ler) veya dikkat mekanizmaları (attention) daha büyük ve gerçek dünya veri setlerinde önemli performans artışları sağlayabilir.

## Kaynaklar (References)

* Temel Tekrarlayan Sinir Ağları (RNN) Mimarisi.
* Uzun-Kısa Süreli Bellek (LSTM) Ağları Mimarisi ve Çalışma Prensibi.
* Zaman Boyunca Geri Yayılım (Backpropagation Through Time - BPTT) Algoritması.
* Stokastik Gradyan Azaltma (SGD) ve Adam Optimizasyon Algoritmaları.
* İkili Çapraz Entropi (Binary Cross-Entropy) Kayıp Fonksiyonu.
* Sigmoid ve Tanh Aktivasyon Fonksiyonları.
* Gradient Clipping (Gradyan Kırpma) Tekniği.
* PyTorch Derin Öğrenme Kütüphanesi (torch.org).
* NumPy Bilimsel Hesaplama Kütüphanesi (numpy.org).

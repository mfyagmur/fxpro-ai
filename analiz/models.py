from django.db import models
from django.contrib.auth.models import User

class Varlik(models.Model):
    KATEGORILER = (
        ('EMTIA', 'Emtia (Altın, Petrol)'),
        ('FOREX', 'Forex (Döviz)'),
        ('KRIPTO', 'Kripto Para'),
        ('BORSA_TR', 'BIST 100 Hisseleri'),
        ('BORSA_US', 'ABD Borsası'),
    )

    isim = models.CharField(max_length=50, verbose_name="Varlık Adı") # Örn: Altın (Ons)
    sembol = models.CharField(max_length=20, unique=True, verbose_name="Yahoo Kodu") # Örn: GC=F
    kategori = models.CharField(max_length=20, choices=KATEGORILER, default='FOREX')
    aktif = models.BooleanField(default=True, verbose_name="Listede Göster")

    def __str__(self):
        return f"{self.isim} ({self.sembol})"

    class Meta:
        verbose_name = "Finansal Varlık"
        verbose_name_plural = "Finansal Varlıklar"

class Portfoy(models.Model):
    varlik = models.ForeignKey(Varlik, on_delete=models.CASCADE, verbose_name="Varlık")
    miktar = models.FloatField(verbose_name="Eldeki Miktar (Adet/Lot)")
    maliyet = models.FloatField(verbose_name="Ortalama Maliyet ($/TL)")
    tarih = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.varlik.isim} - {self.miktar} Adet"
    
    class Meta:
        verbose_name = "Portföy Kaydı"
        verbose_name_plural = "Portföyüm"

class Portfoy(models.Model):
    # YENİ: Her kaydı bir kullanıcıya bağlıyoruz
    kullanici = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Kullanıcı", default=1) 
    
    varlik = models.ForeignKey(Varlik, on_delete=models.CASCADE, verbose_name="Varlık")
    miktar = models.FloatField(verbose_name="Eldeki Miktar (Adet/Lot)")
    maliyet = models.FloatField(verbose_name="Ortalama Maliyet ($/TL)")
    tarih = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.kullanici.username} - {self.varlik.isim}"
    
    class Meta:
        verbose_name = "Portföy Kaydı"
        verbose_name_plural = "Portföyüm"
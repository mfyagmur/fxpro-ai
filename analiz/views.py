from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from prophet import Prophet
from .models import Varlik, Portfoy
from .utils import telegram_gonder
from sklearn.metrics import mean_absolute_error
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor # YENİ: Daha Güçlü Beyin
from textblob import TextBlob
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# --- 1. HARİTA VE AYARLAR ---

def sembol_donustur_tv(yahoo_sembol):
    """
    Yahoo Finance kodlarını TradingView widget formatına çevirir.
    Hataları önlemek için en popüler borsalar eklendi.
    """
    harita = {
        # EMTİALAR
        "GC=F": "OANDA:XAUUSD",       
        "SI=F": "TVC:SILVER",         
        "CL=F": "TVC:USOIL",          
        "NG=F": "TVC:NATURALGAS",     
        
        # FOREX
        "EURUSD=X": "FX:EURUSD",
        "JPY=X": "FX:USDJPY",         
        "GBPUSD=X": "FX:GBPUSD",
        "TRY=X": "FX:USDTRY",
        
        # KRİPTO (Binance en stabil veridir)
        "BTC-USD": "BINANCE:BTCUSDT", 
        "ETH-USD": "BINANCE:ETHUSDT",
        "SOL-USD": "BINANCE:SOLUSDT",
        "AVAX-USD": "BINANCE:AVAXUSDT",
        
        # BORSA ABD
        "^GSPC": "OANDA:SPX500USD",   
        "TSLA": "NASDAQ:TSLA",
        "AAPL": "NASDAQ:AAPL",
        "GOOG": "NASDAQ:GOOGL", # Google kodu GOOGL'dir
        "MSFT": "NASDAQ:MSFT",
        
        # BORSA İSTANBUL (BIST) - KRİTİK DÜZELTME
        # TradingView widget'ında BIST verileri bazen gecikmeli gelir.
        "THYAO.IS": "THYAO", 
        "ASELS.IS": "ASELS",
        "GARAN.IS": "GARAN",
        "AKBNK.IS": "AKBNK",
        "EREGL.IS": "EREGL",
        "XU100.IS": "XU100", 
    }
    
    # Eğer listede yoksa, Yahoo kodundaki .IS gibi uzantıları temizleyip şansımızı deneyelim
    if yahoo_sembol not in harita:
        temiz_sembol = yahoo_sembol.replace(".IS", "").replace("=F", "").replace("=X", "")
        return temiz_sembol
        
    return harita.get(yahoo_sembol, yahoo_sembol)

def kategoriye_gore_haber_ayarlari(sembol):
    """Sembole göre hangi haberleri tarayacağımızı belirler."""
    
    # Varsayılan (Genel Finans)
    rss_urls = ["https://finance.yahoo.com/news/rssindex"]
    anahtar_kelimeler = ["market", "economy", "fed", "rate", "inflation"]
    
    if "BTC" in sembol or "ETH" in sembol or "SOL" in sembol:
        # KRİPTO MODU
        rss_urls = ["https://www.coindesk.com/arc/outboundfeeds/rss/"]
        anahtar_kelimeler = ["crypto", "bitcoin", "ethereum", "blockchain", "coin"]
        
    elif ".IS" in sembol or "TRY" in sembol:
        # TÜRKİYE / BIST MODU (TRT Haber Ekonomi)
        rss_urls = ["https://www.trthaber.com/xml_mobile_ekonomi.rss"]
        anahtar_kelimeler = ["borsa", "bist", "dolar", "enflasyon", "faiz", "tcmb", "aselsan", "thy"]
        
    elif "GC=F" in sembol or "CL=F" in sembol:
        # EMTİA MODU
        anahtar_kelimeler.extend(["gold", "oil", "energy", "commodity", "crude"])
        
    return rss_urls, anahtar_kelimeler

# --- 2. GÜÇLENDİRİLMİŞ MOTORLAR ---

def veri_getir(sembol="GC=F", periyot="1y"):
    try:
        # Veri setini biraz daha genişlettik (Daha iyi öğrenmesi için)
        df = yf.download(sembol, period=periyot, interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def rsi_hesapla(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def yapay_zeka_tahmini_v3(df):
    try:
        df_prophet = df.reset_index()[['Date', 'Close']].copy()
        df_prophet.columns = ['ds', 'y']
        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)

        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(df_prophet)
        
        # --- BACKTEST KISMI (YENİ) ---
        # Geçmiş son 3 günün verisini alıp kıyaslayalım
        gecmis_df = df_prophet.tail(3).copy()
        gecmis_tahmin = model.predict(gecmis_df)
        
        karni = []
        for index, row in gecmis_df.iterrows():
            gercek = row['y']
            # Aynı tarihe denk gelen tahmini bul
            tahmin_row = gecmis_tahmin[gecmis_tahmin['ds'] == row['ds']]
            if not tahmin_row.empty:
                tahmin_deger = tahmin_row.iloc[0]['yhat']
                fark_yuzde = abs(gercek - tahmin_deger) / gercek * 100
                basari = 100 - fark_yuzde
                
                karni.append({
                    'tarih': row['ds'].strftime('%d %b'), # Örn: 05 Jan
                    'gercek': round(gercek, 2),
                    'tahmin': round(tahmin_deger, 2),
                    'sapma': round(fark_yuzde, 2)
                })
        # -----------------------------

        # Gelecek Tahmini
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        
        yarin_tahmin = forecast.iloc[-1]['yhat']
        alt_sinir = forecast.iloc[-1]['yhat_lower']
        ust_sinir = forecast.iloc[-1]['yhat_upper']
        
        # Basit genel skor (Son günün başarısı)
        genel_skor = karni[-1]['sapma'] if karni else 0
        basari_skoru = max(0, 100 - genel_skor)

        return yarin_tahmin, alt_sinir, ust_sinir, basari_skoru, karni # 5 DEĞER DÖNDÜRÜYORUZ
        
    except Exception as e:
        print(f"Prophet Hatası: {e}")
        return 0, 0, 0, 0, []

def haber_analizi_akilli(sembol):
    rss_urls, keywords = kategoriye_gore_haber_ayarlari(sembol)
    toplam_puan = 0
    sayac = 0
    
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:7]: # İlk 7 haberi tara
                baslik = entry.title.lower()
                
                # FİLTRE: Konuyla alakalı mı?
                # Eğer Türk haberi ise textblob (ingilizce) yerine basit kelime kontrolü yapalım
                if ".IS" in sembol or "TRY" in sembol:
                    # Türkçe basit analiz (Negatif kelime listesi)
                    if any(k in baslik for k in ["düşüş", "zarar", "kriz", "savaş", "gergin"]):
                        toplam_puan -= 0.5
                        sayac += 1
                    elif any(k in baslik for k in ["rekor", "zirve", "yükseliş", "kazanç", "büyüme"]):
                        toplam_puan += 0.5
                        sayac += 1
                else:
                    # İngilizce Analiz (TextBlob)
                    if any(k in baslik for k in keywords):
                        analiz = TextBlob(entry.title)
                        puan = analiz.sentiment.polarity
                        if abs(puan) > 0.05:
                            toplam_puan += puan
                            sayac += 1
        except:
            continue
            
    return toplam_puan / sayac if sayac > 0 else 0

# --- 3. VIEW FONKSİYONLARI ---

def dashboard(request):
    # 1. Varlık Listesi
    varliklar = Varlik.objects.filter(aktif=True)
    varsayilan = varliklar.first() if varliklar.exists() else None
    
    # 2. PORTFÖY HESAPLAMALARI (YENİ KISIM) 💰
    portfoy_kayitlari = []
    
    # Eğer kullanıcı giriş yapmışsa sadece KENDİ verilerini çek
    if request.user.is_authenticated:
        portfoy_kayitlari = Portfoy.objects.filter(kullanici=request.user)
    else:
        # Giriş yapmamışsa boş liste (Güvenlik)
        portfoy_kayitlari = Portfoy.objects.none()
    
    toplam_deger = 0      # Şu anki toplam paramız ($)
    toplam_maliyet = 0    # Cepten çıkan toplam para ($)
    portfoy_detay = []    # Ekrana basacağımız liste
    
    for kayit in portfoy_kayitlari:
        # Her varlığın güncel fiyatını çek
        df = veri_getir(kayit.varlik.sembol, periyot="5d")
        if not df.empty:
            guncel_fiyat = float(df['Close'].iloc[-1])
            
            # Matematik: (Adet * Fiyat)
            anlik_tutar = guncel_fiyat * kayit.miktar
            maliyet_tutar = kayit.maliyet * kayit.miktar
            
            kar_zarar = anlik_tutar - maliyet_tutar
            kar_zarar_yuzde = (kar_zarar / maliyet_tutar) * 100 if maliyet_tutar > 0 else 0
            
            toplam_deger += anlik_tutar
            toplam_maliyet += maliyet_tutar
            
            portfoy_detay.append({
                'isim': kayit.varlik.isim,
                'adet': kayit.miktar,
                'maliyet': kayit.maliyet,
                'guncel': round(guncel_fiyat, 2),
                'kar_zarar': round(kar_zarar, 2),
                'kar_zarar_yuzde': round(kar_zarar_yuzde, 2),
                'durum_renk': 'text-green-400' if kar_zarar > 0 else 'text-red-500'
            })
    
    toplam_kar = toplam_deger - toplam_maliyet
    genel_durum_renk = "text-green-400" if toplam_kar > 0 else "text-red-500"

    context = {
        'varliklar': varliklar,
        'secili_sembol': varsayilan.sembol if varsayilan else 'GC=F',
        # Portföy verilerini HTML'e gönderiyoruz
        'portfoy_var': portfoy_kayitlari.exists(),
        'toplam_deger': round(toplam_deger, 2),
        'toplam_kar': round(toplam_kar, 2),
        'genel_durum_renk': genel_durum_renk,
        'portfoy_detay': portfoy_detay
    }
    
    return render(request, 'analiz/dashboard.html', context)

def varlik_verisi_getir(request):
    sembol = request.GET.get('sembol', 'GC=F')
    
    # 1. Veri
    df = veri_getir(sembol)
    if df.empty: return JsonResponse({'error': 'Veri yok'}, status=400)

    # 2. Hesaplamalar
    son_fiyat = float(df['Close'].iloc[-1])
    onceki_fiyat = float(df['Close'].iloc[-2])
    degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
    
    df['RSI'] = rsi_hesapla(df['Close'])
    rsi = float(df['RSI'].iloc[-1])
    
    # YENİ AI MODELİNİ KULLAN
    tahmin, alt_sinir, ust_sinir, basari_skoru, karne = yapay_zeka_tahmini_v3(df)
    karne_html = ""
    for k in karne:
        renk = "text-green-400" if k['sapma'] < 2 else "text-yellow-400"
        karne_html += f"""
        <tr class="border-b border-gray-700/50">
            <td class="py-2 text-gray-400">{k['tarih']}</td>
            <td class="py-2 text-right text-white">${k['gercek']}</td>
            <td class="py-2 text-right text-purple-300">${k['tahmin']}</td>
            <td class="py-2 text-right {renk}">% {k['sapma']}</td>
        </tr>
        """
        
    if tahmin == 0:
        tahmin = son_fiyat 
        
    fark = tahmin - son_fiyat
    # Haber Analizi   
    haber_skoru = haber_analizi_akilli(sembol)

    # 3. Skorlama
    skor = 0
    if tahmin > son_fiyat: skor += 1
    if rsi < 30: skor += 2
    elif rsi > 70: skor -= 2
    
    # Haber Puanı Etkisi (Daha hassas)
    if haber_skoru > 0.1: skor += 1.5
    elif haber_skoru < -0.1: skor -= 1.5
    
    karar_metni, karar_class = "BEKLE", "text-yellow-400"
    if skor >= 2.5: 
        karar_metni, karar_class = "GÜÇLÜ AL 🚀", "text-green-400"
        mesaj = f"🚨 *SİNYAL ALARMI* 🚨\n\n📈 *Varlık:* {sembol}\n💡 *Karar:* GÜÇLÜ AL\n🎯 *Hedef:* {tahmin:.2f}\n💰 *Anlık:* {son_fiyat:.2f}"
        telegram_gonder(mesaj)
    elif skor <= -2.5: 
        karar_metni, karar_class = "GÜÇLÜ SAT 🔻", "text-red-500"
        mesaj = f"🚨 *SİNYAL ALARMI* 🚨\n\n📉 *Varlık:* {sembol}\n💡 *Karar:* GÜÇLÜ SAT\n🔻 *Risk:* Düşüş Beklentisi\n💰 *Anlık:* {son_fiyat:.2f}"
        telegram_gonder(mesaj)
    elif skor > 0.5: karar_metni, karar_class = "ALIM YÖNLÜ 📈", "text-blue-400"
    elif skor < -0.5: karar_metni, karar_class = "SATIŞ YÖNLÜ 📉", "text-orange-400"

    # 4. Yanıt
    data = {
        'fiyat': f"{son_fiyat:.2f}",
        'degisim': f"{degisim:.2f}",
        'degisim_pozitif': degisim > 0,
        'karar': karar_metni,
        'karar_class': karar_class,
        'rsi': f"{rsi:.0f}",
        'tahmin': f"{tahmin:.2f}",
        'basari_skoru': f"%{basari_skoru:.1f}", # Başarı Skoru
        'karne_html': karne_html, # Geçmiş tahmin karne tablosu
        'aralik': f"{alt_sinir:.2f} - {ust_sinir:.2f}", # YENİ: Güven Aralığı
        'tahmin_yonu': "YÜKSELİŞ 📈" if fark > 0 else "DÜŞÜŞ 📉",
        'haber_durumu': "POZİTİF 😊" if haber_skoru > 0 else "NEGATİF 😟" if haber_skoru < 0 else "NÖTR 😐",
        'tradingview_symbol': sembol_donustur_tv(sembol)
    }
    
    return JsonResponse(data)

def detayli_grafik_getir(request):
    sembol = request.GET.get('sembol', 'GC=F')
    
    # 1. Veriyi Getir
    df = veri_getir(sembol)
    if df.empty: return JsonResponse({'error': 'Veri yok'})

    # Data Type Fix (Sayıya çevirme garantisi)
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # 2. İndikatörler
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # 3. AI Tahmini (5 Değeri de alıyoruz)
    tahmin, alt, ust, basari_skoru, karne = yapay_zeka_tahmini_v3(df)
    
    # 4. GRAFİK ÇİZİMİ (Burası eksikti, şimdi ekledik)
    fig = make_subplots(rows=1, cols=1)
    
    # A) Fiyat Mumları
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Fiyat',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # B) Trend (SMA)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_20'],
        line=dict(color='#fbbf24', width=2),
        name='Trend (Ortalama)',
        opacity=0.8
    ))
    
    # C) Güven Aralığı (Tünel)
    tahmin_tarihi = df.index[-1] + pd.Timedelta(days=1)
    fig.add_trace(go.Scatter(
        x=[tahmin_tarihi, tahmin_tarihi],
        y=[alt, ust],
        mode='lines',
        line=dict(color='#d946ef', width=4, dash='dot'),
        name='Güven Aralığı'
    ))

    # D) AI Yıldızı
    fig.add_trace(go.Scatter(
        x=[tahmin_tarihi], y=[tahmin],
        mode='markers+text',
        marker=dict(color='#d946ef', size=18, symbol='star'),
        text=[f"HEDEF<br>{tahmin:.2f}"],
        textposition="top center",
        textfont=dict(color='#d946ef', size=14, family="Arial Black"),
        name='AI Tahmini'
    ))

    # Layout Ayarları
    fig.update_layout(
        template='plotly_dark',
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20, 20, 30, 0.5)',
        title=dict(text=f"{sembol} DETAYLI TEKNİK ANALİZ", x=0.02, y=0.98),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)', rangeslider_visible=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
        modebar=dict(remove=['zoom', 'pan', 'select', 'lasso2d', 'logo'], bgcolor='rgba(0,0,0,0)')
    )
    
    # 5. TABLO OLUŞTURMA (Backtest Karnesi)
    karne_html = '<table class="w-full text-sm text-left"><thead class="bg-gray-700/30 text-xs uppercase text-gray-400"><tr><th class="px-4 py-2">Tarih</th><th class="px-4 py-2 text-right">Gerçek</th><th class="px-4 py-2 text-right">AI Tahmin</th><th class="px-4 py-2 text-right">Hata</th></tr></thead><tbody>'
    
    for k in karne:
        renk = "text-green-400" if k['sapma'] < 2 else "text-red-400"
        karne_html += f'<tr class="border-b border-gray-700/20"><td class="px-4 py-2 text-gray-400">{k["tarih"]}</td><td class="px-4 py-2 text-right">${k["gercek"]}</td><td class="px-4 py-2 text-right text-purple-300">${k["tahmin"]}</td><td class="px-4 py-2 text-right {renk}">%{k["sapma"]}</td></tr>'
    
    karne_html += '</tbody></table>'

    # 6. Render
    grafik_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn', config={'displayModeBar': True, 'displaylogo': False})
    
    return JsonResponse({'grafik_html': grafik_html, 'karne_html': karne_html})

def kayit_ol(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user) # Kayıt olunca otomatik giriş yap
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'analiz/register.html', {'form': form})

"""def detayli_grafik_getir(request):
    sembol = request.GET.get('sembol', 'GC=F')
    df = veri_getir(sembol)
    
    if df.empty: return JsonResponse({'error': 'Veri yok'})

    # Hesaplamalar
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    tahmin = yapay_zeka_tahmini_v2(df)
    
    # Grafik Çiz
    fig = make_subplots(rows=1, cols=1)
    
    # Mumlar
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'))
    
    # SMA
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=2), name='Trend (SMA 20)'))
    
    # AI Tahmin Noktası (MOR YILDIZ)
    tahmin_tarihi = df.index[-1] + pd.Timedelta(days=1)
    fig.add_trace(go.Scatter(
        x=[tahmin_tarihi], y=[tahmin],
        mode='markers+text',
        marker=dict(color='purple', size=15, symbol='star'),
        text=[f"HEDEF: {tahmin:.2f}"],
        textposition="top center",
        name='AI Tahmini'
    ))

    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)', # Şeffaf arka plan
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    grafik_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    return JsonResponse({'grafik_html': grafik_html})"""
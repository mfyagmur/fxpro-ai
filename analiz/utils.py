import requests

# 1. TOKEN (BotFather'dan aldığın)
TELEGRAM_TOKEN = "8239847394:AAG5tPPRPVhHS-_UqozVk_3AeKlVdpZ7qtE"

# 2. CHAT ID (Senin numaranı buraya yazdım)
CHAT_ID = "6579457404" 

def telegram_gonder(mesaj):
    """
    Belirtilen mesajı Telegram botu üzerinden cebine gönderir.
    """
    if not CHAT_ID:
        print("⚠️ Telegram Chat ID eksik! Mesaj gönderilemedi.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": mesaj,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"✅ Mesaj gönderildi: {mesaj}")
        else:
            print(f"❌ Mesaj hatası: {response.text}")
    except Exception as e:
        print(f"Bağlantı hatası: {e}")
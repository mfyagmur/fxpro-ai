from django.contrib import admin
from .models import Varlik, Portfoy

@admin.register(Varlik)
class VarlikAdmin(admin.ModelAdmin):
    list_display = ('isim', 'sembol', 'kategori', 'aktif')
    list_filter = ('kategori', 'aktif')
    search_fields = ('isim', 'sembol')

@admin.register(Portfoy)
class PortfoyAdmin(admin.ModelAdmin):
    list_display = ('varlik', 'miktar', 'maliyet', 'kar_zarar_durumu')
    
    def kar_zarar_durumu(self, obj):
        return "Canlı hesaplanacak"
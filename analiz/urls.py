from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('api/get_data/', views.varlik_verisi_getir, name='get_data'), # YENİ API
    path('api/get_details/', views.detayli_grafik_getir, name='get_details'), # YENİ API
    path('giris/', auth_views.LoginView.as_view(template_name='analiz/login.html'), name='login'),
    path('cikis/', auth_views.LogoutView.as_view(next_page='dashboard'), name='logout'),
    path('kayit/', views.kayit_ol, name='register'),
]
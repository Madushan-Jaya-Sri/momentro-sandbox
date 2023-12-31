from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    
    path('', include('m_suite.urls')),
    
    
]

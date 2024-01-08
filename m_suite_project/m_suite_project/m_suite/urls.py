from django.urls import path
from .views import website_keyword, extract_links
from m_suite import views
urlpatterns = [
    #path('', demo, name='index'),
    #path('', views.index, name='index'),
    
    path('', views.overview, name='overview'),
    
    path('website-keyword', views.extract_links, name='extract_links'),
    path('history', views.history, name='history'),
    path('youtube', views.proceed_yt_url, name='proceed_yt_url'),
    path('youtube/download-yt-dataset/', views.download_yt_comments, name='download_yt_comments'),
    path('website-keyword/download-dataset/', views.download_dataset, name='download_dataset'),
    path('overview', views.overview, name='overview'),
    path('sentiment_analysis', views.sentiment_analysis, name='sentiment_analysis'),
    path('brand_authority', views.brand_authority, name='brand_authority'),
    path('brand_personality', views.brand_personality, name='brand_personality'),
    path('profile_analyzer', views.profile_analyzer, name='profile_analyzer'),
    path('website_keyword', views.website_keyword, name='website_keyword'),

    

]


    #path('my-view/', views.my_view, name='your_view_name'),


    
   
    



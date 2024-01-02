# models.py
from django.db import models

class keyword_count_data(models.Model):
    Keyword = models.CharField(max_length=150)
    Count = models.IntegerField()
    Url = models.CharField(max_length=200)

    def __str__(self):
        return f"{self.Keyword} - {self.Count}"

class youtube_comments(models.Model):
    
    Channel_Name = models.CharField(max_length=200)
    Comments = models.TextField()  # Use TextField for potentially longer comments

    def __str__(self):
        return f"{self.Channel_Name} - {self.Comments}"

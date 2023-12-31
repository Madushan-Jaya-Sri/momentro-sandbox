# models.py
from django.db import models

class keyword_count_data(models.Model):
    Keyword = models.CharField(max_length=150)
    Count = models.IntegerField()
    Url = models.CharField(max_length=200)

    def __str__(self):
        return f"{self.Keyword} - {self.Count}"

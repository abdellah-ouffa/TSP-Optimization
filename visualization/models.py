from django.db import models


class TSPResult(models.Model):
    algorithm = models.CharField(max_length=50)
    tour = models.TextField()
    total_distance = models.FloatField()
    home_city = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.algorithm} - {self.created_at}"

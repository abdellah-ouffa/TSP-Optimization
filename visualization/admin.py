from django.contrib import admin
from django.contrib.admin import AdminSite
from .models import TSPResult


class TSPAdminSite(AdminSite):
    site_header = "TSP Admin"
    site_title = "TSP Admin Portal"
    index_title = "Welcome to TSP Admin"


tsp_admin_site = TSPAdminSite(name="tsp_admin")


@admin.register(TSPResult, site=tsp_admin_site)
class TSPResultAdmin(admin.ModelAdmin):
    list_display = ("algorithm", "total_distance", "home_city", "created_at")
    list_filter = ("algorithm", "created_at")
    search_fields = ("home_city",)
    ordering = ("-created_at",)

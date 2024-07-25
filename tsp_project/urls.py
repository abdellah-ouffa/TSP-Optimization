from django.urls import path, include
from visualization.admin import tsp_admin_site

urlpatterns = [
    path("admin/", tsp_admin_site.urls),
    path("", include("visualization.urls")),
]

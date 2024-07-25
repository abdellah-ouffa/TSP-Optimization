from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path(
        "data/nearest_neighbor/", views.nearest_neighbor_view, name="nearest_neighbor"
    ),
    path(
        "data/genetic_algorithm/",
        views.genetic_algorithm_view,
        name="genetic_algorithm",
    ),
    path("data/two_opt/", views.two_opt_view, name="two_opt"),
    path(
        "data/simulated_annealing/",
        views.simulated_annealing_view,
        name="simulated_annealing",
    ),
    path("data/tsp-results/", views.get_tsp_results, name="tsp_results"),
    path("data-analysis/", views.data_analysis_view, name="data_analysis"),
]

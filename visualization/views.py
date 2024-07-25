from django.utils import timezone
from django.shortcuts import render
from django.http import JsonResponse
from random import shuffle, choice, random
import csv
import random
import numpy as np
import pandas as pd
import logging
from .models import TSPResult
import math
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


def load_distances(file_path):
    distances = {}
    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            city1, city2, distance = row
            distance = float(distance)
            distances[(city1, city2)] = distance
            distances[(city2, city1)] = distance
    return distances


def data_analysis_view(request):
    data = TSPResult.objects.all().values()

    df = pd.DataFrame(data)

    df_avg = df.groupby(["home_city", "algorithm"]).total_distance.mean().reset_index()

    plt.figure(figsize=(14, 8))
    sns.barplot(x="home_city", y="total_distance", hue="algorithm", data=df_avg)
    plt.title("Average Total Distance of TSP Algorithms by Starting City")
    plt.xlabel("Starting City")
    plt.ylabel("Average Total Distance")
    plt.xticks(rotation=45)
    plt.legend(title="Algorithm")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png)
    graphic = graphic.decode("utf-8")

    return render(request, "visualization/data_analysis.html", {"graphic": graphic})


def nearest_neighbor(cities, distances, start_city):
    unvisited = set(cities)
    tour = [start_city]
    unvisited.remove(start_city)
    while unvisited:
        current = tour[-1]
        next_city = min(unvisited, key=lambda city: distances[(current, city)])
        tour.append(next_city)
        unvisited.remove(next_city)
    tour.append(tour[0])
    return tour


def calculate_distance(tour, distances):
    return sum(distances[(tour[i], tour[i + 1])] for i in range(len(tour) - 1))


def genetic_algorithm(
    cities, distances, population_size=100, generations=1000, mutation_rate=0.01
):
    def create_route(cities):
        return random.sample(cities, len(cities))

    def initial_population(population_size, cities):
        return [create_route(cities) for _ in range(population_size)]

    def rank_routes(population):
        fitness_results = {}
        for i, route in enumerate(population):
            fitness_results[i] = 1 / float(
                calculate_distance(route + [route[0]], distances)
            )
        return sorted(fitness_results.items(), key=lambda x: x[1], reverse=True)

    def selection(pop_ranked, elite_size):
        selection_results = []
        df = pd.DataFrame(np.array(pop_ranked), columns=["Index", "Fitness"])
        df["cum_sum"] = df.Fitness.cumsum()
        df["cum_perc"] = 100 * df.cum_sum / df.Fitness.sum()

        for i in range(elite_size):
            selection_results.append(pop_ranked[i][0])
        for _ in range(len(pop_ranked) - elite_size):
            pick = 100 * random.random()
            for i in range(len(pop_ranked)):
                if pick <= df.iat[i, 3]:
                    selection_results.append(pop_ranked[i][0])
                    break
        return selection_results

    def mating_pool(population, selection_results):
        return [population[i] for i in selection_results]

    def breed(parent1, parent2):
        child = []
        childP1 = []
        childP2 = []

        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))

        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        for i in range(startGene, endGene):
            childP1.append(parent1[i])

        childP2 = [item for item in parent2 if item not in childP1]

        child = childP1 + childP2
        return child

    def breed_population(matingpool, elite_size):
        children = []
        length = len(matingpool) - elite_size
        pool = random.sample(matingpool, len(matingpool))

        for i in range(elite_size):
            children.append(matingpool[i])

        for i in range(length):
            child = breed(pool[i], pool[len(matingpool) - i - 1])
            children.append(child)
        return children

    def mutate(individual, mutation_rate):
        for swapped in range(len(individual)):
            if random.random() < mutation_rate:
                swapWith = int(random.random() * len(individual))

                city1 = individual[swapped]
                city2 = individual[swapWith]

                individual[swapped] = city2
                individual[swapWith] = city1
        return individual

    def mutate_population(population, mutation_rate):
        return [mutate(ind, mutation_rate) for ind in population]

    def next_generation(current_gen, elite_size, mutation_rate):
        pop_ranked = rank_routes(current_gen)
        selection_results = selection(pop_ranked, elite_size)
        matingpool = mating_pool(current_gen, selection_results)
        children = breed_population(matingpool, elite_size)
        return mutate_population(children, mutation_rate)

    population = initial_population(population_size, cities)
    elite_size = int(0.1 * population_size)

    for i in range(generations):
        population = next_generation(population, elite_size, mutation_rate)

    best_route_index = rank_routes(population)[0][0]
    best_route = population[best_route_index]

    return best_route + [best_route[0]], calculate_distance(
        best_route + [best_route[0]], distances
    )


def two_opt(cities, distances, max_iterations=1000):
    def swap_2opt(route, i, k):
        return route[:i] + route[i : k + 1][::-1] + route[k + 1 :]

    best_route = cities + [cities[0]]
    best_distance = calculate_distance(best_route, distances)
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        for i in range(1, len(cities) - 2):
            for k in range(i + 1, len(cities)):
                new_route = swap_2opt(best_route, i, k)
                new_distance = calculate_distance(new_route, distances)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
        iterations += 1

    return best_route, best_distance


def simulated_annealing(
    cities, distances, temperature=10000, cooling_rate=0.995, max_iterations=10000
):
    def get_neighbor(route):
        i, j = sorted(random.sample(range(1, len(route) - 1), 2))
        return route[:i] + route[i : j + 1][::-1] + route[j + 1 :]

    current_route = cities + [cities[0]]
    current_distance = calculate_distance(current_route, distances)
    best_route = current_route
    best_distance = current_distance

    for _ in range(max_iterations):
        if temperature < 0.1:
            break
        neighbor_route = get_neighbor(current_route)
        neighbor_distance = calculate_distance(neighbor_route, distances)

        if neighbor_distance < current_distance:
            current_route = neighbor_route
            current_distance = neighbor_distance
            if current_distance < best_distance:
                best_route = current_route
                best_distance = current_distance
        elif random.random() < math.exp(
            (current_distance - neighbor_distance) / temperature
        ):
            current_route = neighbor_route
            current_distance = neighbor_distance

        temperature *= cooling_rate

    return best_route, best_distance


file_path = r"C:\Users\ouffa\OneDrive\Desktop\SPRING SEMESTER LECTURES\computer graphics optimization\FINAl_PROJECT\city_distances.csv"
distances = load_distances(file_path)

cities = [
    "Tokyo",
    "New York",
    "Los Angeles",
    "Paris",
    "London",
    "Beijing",
    "Moscow",
    "Sydney",
    "Rio de Janeiro",
    "Cape Town",
    "Rabat",
    "Berlin",
    "Mumbai",
    "Mexico City",
    "Toronto",
    "Buenos Aires",
]


def index(request):
    return render(request, "visualization/index.html")


def nearest_neighbor_view(request):
    logger.info("Nearest Neighbor view called")
    try:
        start_city = choice(cities)
        cities_copy = cities.copy()
        shuffle(cities_copy)
        tour = nearest_neighbor(cities_copy, distances, start_city)
        total_distance = calculate_distance(tour, distances)

        result = TSPResult.objects.create(
            algorithm="Nearest Neighbor",
            tour=str(tour),
            total_distance=total_distance,
            home_city=start_city,
        )
        result.save()

        response_data = {
            "tour": tour,
            "total_distance": total_distance,
            "home_city": start_city,
        }
        logger.info("Nearest Neighbor response: %s", response_data)
        return JsonResponse(response_data)
    except Exception as e:
        logger.error("Error in Nearest Neighbor view: %s", str(e))
        return JsonResponse({"error": "An error occurred"}, status=500)


def genetic_algorithm_view(request):
    logger.info("Genetic Algorithm view called")
    try:
        population_size = int(request.GET.get("population_size", 100))
        generations = int(request.GET.get("generations", 1000))
        mutation_rate = float(request.GET.get("mutation_rate", 0.01))
        tour, total_distance = genetic_algorithm(
            cities, distances, population_size, generations, mutation_rate
        )

        result = TSPResult.objects.create(
            algorithm="Genetic Algorithm",
            tour=str(tour),
            total_distance=total_distance,
            home_city=tour[0],
        )
        result.save()

        response_data = {
            "tour": tour,
            "total_distance": total_distance,
            "home_city": tour[0],
        }
        logger.info("Genetic Algorithm response: %s", response_data)
        return JsonResponse(response_data)
    except (ValueError, TypeError) as e:
        logger.error("Error in Genetic Algorithm view: %s", str(e))
        return JsonResponse({"error": "Invalid parameters"}, status=400)


def two_opt_view(request):
    logger.info("2-opt Algorithm view called")
    try:
        max_iterations = int(request.GET.get("max_iterations", 1000))
        start_city = choice(cities)
        cities_copy = cities.copy()
        shuffle(cities_copy)
        tour, total_distance = two_opt(cities_copy, distances, max_iterations)

        result = TSPResult.objects.create(
            algorithm="2-opt",
            tour=str(tour),
            total_distance=total_distance,
            home_city=start_city,
        )
        result.save()

        response_data = {
            "tour": tour,
            "total_distance": total_distance,
            "home_city": start_city,
        }
        logger.info("2-opt Algorithm response: %s", response_data)
        return JsonResponse(response_data)
    except Exception as e:
        logger.error("Error in 2-opt Algorithm view: %s", str(e))
        return JsonResponse({"error": "An error occurred"}, status=500)


def simulated_annealing_view(request):
    logger.info("Simulated Annealing Algorithm view called")
    try:
        temperature = float(request.GET.get("temperature", 10000))
        cooling_rate = float(request.GET.get("cooling_rate", 0.995))
        max_iterations = int(request.GET.get("max_iterations", 10000))
        start_city = choice(cities)
        cities_copy = cities.copy()
        shuffle(cities_copy)
        tour, total_distance = simulated_annealing(
            cities_copy, distances, temperature, cooling_rate, max_iterations
        )

        result = TSPResult.objects.create(
            algorithm="Simulated Annealing",
            tour=str(tour),
            total_distance=total_distance,
            home_city=start_city,
        )
        result.save()

        response_data = {
            "tour": tour,
            "total_distance": total_distance,
            "home_city": start_city,
        }
        logger.info("Simulated Annealing Algorithm response: %s", response_data)
        return JsonResponse(response_data)
    except Exception as e:
        logger.error("Error in Simulated Annealing Algorithm view: %s", str(e))
        return JsonResponse({"error": "An error occurred"}, status=500)


def get_tsp_results(request):
    results = TSPResult.objects.all().order_by("-created_at")
    data = [
        {
            "algorithm": result.algorithm,
            "tour": eval(result.tour),
            "total_distance": result.total_distance,
            "home_city": result.home_city,
            "created_at": result.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            # "created_at": timezone.localtime(result.created_at).isoformat(),
        }
        for result in results
    ]
    return JsonResponse(data, safe=False)

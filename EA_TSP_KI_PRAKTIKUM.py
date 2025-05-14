import numpy as np
import random
import gzip
import math
import matplotlib.pyplot as plt


def read_tsp_gz(file_path):
    with gzip.open(file_path, 'rt') as f:
        lines = f.readlines()

    coords = []
    in_section = False
    for line in lines:
        if 'NODE_COORD_SECTION' in line:
            in_section = True
            continue
        if 'EOF' in line:
            break
        if in_section:
            parts = line.strip().split()
            if len(parts) >= 3:
                x, y = float(parts[1]), float(parts[2])
                coords.append((x, y))
    return coords

def compute_distance_matrix(coords):
    n = len(coords)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 9999
            else:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                matrix[i][j] = round(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
    return matrix

def evaluate_route(route, matrix):
    total = 0
    for i in range(len(route)):
        total += matrix[route[i]][route[(i + 1) % len(route)]]
    return total

def create_initial_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

def mutate(route):
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]
    return route

def crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = parent1[a:b]
    fill_values = [city for city in parent2 if city not in child]
    pointer = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill_values[pointer]
            pointer += 1
    return child

def plot_route(route, coords):
    x = [coords[i][0] for i in route + [route[0]]]
    y = [coords[i][1] for i in route + [route[0]]]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'bo-')
    plt.title("Best Route Found")
    plt.xlabel("X")
    plt.ylabel("Y")
    for i, (x0, y0) in enumerate(coords):
        plt.text(x0, y0, str(i), fontsize=8)
    plt.grid(True)
    plt.show()

#coords = read_tsp_gz("eil101.tsp.gz")
#distance_matrix = compute_distance_matrix(coords)

#best_route = evolutionary_algorithm(distance_matrix)
#print("✅ Best route:", best_route)
#print("🧮 Total distance:", evaluate_route(best_route, distance_matrix))

#plot_route(best_route, coords)

mutation_rates = [0.1, 0.3, 0.6]

for rate in mutation_rates:
    print(f"\n🧪 Testlauf mit Mutationsrate: {rate}")
    coords = read_tsp_gz("berlin52.tsp.gz")
    distance_matrix = compute_distance_matrix(coords)

    # Füge die Variable in deinen EA-Aufruf ein
    def evolutionary_algorithm(matrix, generations=100, pop_size=50, mutation_rate=0.2):
        num_cities = len(matrix)
        population = create_initial_population(pop_size, num_cities)

        for gen in range(generations):
            population = sorted(population, key=lambda x: evaluate_route(x, matrix))
            new_population = population[:10]

            while len(new_population) < pop_size:
                parent1, parent2 = random.sample(population[:25], 2)
                child = crossover(parent1, parent2)
                if random.random() < mutation_rate:
                    child = mutate(child)
                new_population.append(child)

            population = new_population
        best = population[0]
        print("→ Beste Route:", best)
        print("→ Distanz:", evaluate_route(best, matrix))
        return best

    evolutionary_algorithm(distance_matrix, mutation_rate=rate)

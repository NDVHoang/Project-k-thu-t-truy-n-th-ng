import numpy as np
import matplotlib.pyplot as plt


def objective_function(x):
    return np.sum(x ** 2)


num_particles = 30
num_dimensions = 2
num_iterations = 50
w = 0.7
c1 = 1.5
c2 = 1.5
bounds = [-10, 10]

position = np.random.uniform(bounds[0], bounds[1], (num_particles, num_dimensions))
velocity = np.random.uniform(-1, 1, (num_particles, num_dimensions))
pbest_position = position.copy()
pbest_value = np.array([objective_function(p) for p in position])
gbest_index = np.argmin(pbest_value)
gbest_position = pbest_position[gbest_index].copy()
gbest_value = pbest_value[gbest_index]

convergence_history = []


for i in range(num_iterations):
    r1 = np.random.rand(num_particles, num_dimensions)
    r2 = np.random.rand(num_particles, num_dimensions)

    velocity = (w * velocity) + (c1 * r1 * (pbest_position - position)) + (c2 * r2 * (gbest_position - position))
    position = position + velocity
    position = np.clip(position, bounds[0], bounds[1])

    for k in range(num_particles):
        fitness = objective_function(position[k])
        if fitness < pbest_value[k]:
            pbest_value[k] = fitness
            pbest_position[k] = position[k].copy()
            if fitness < gbest_value:
                gbest_value = fitness
                gbest_position = position[k].copy()

    convergence_history.append(gbest_value)

    print(f"Iter {i + 1}: Best Value = {gbest_value:.6f}")


plt.plot(convergence_history, color='red', marker='o')
plt.title('Biểu đồ hội tụ PSO (PSO Convergence)')
plt.xlabel('Số vòng lặp (Iteration)')
plt.ylabel('Giá trị tốt nhất (Best Fitness)')
plt.grid(True)
plt.show()
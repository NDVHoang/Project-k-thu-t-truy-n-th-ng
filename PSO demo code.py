import numpy as np
import matplotlib.pyplot as plt  # <--- Thêm thư viện vẽ hình


# 1. Hàm mục tiêu (Sphere)
def objective_function(x):
    return np.sum(x ** 2)


# 2. Tham số
num_particles = 30
num_dimensions = 2
num_iterations = 50
w = 0.7
c1 = 1.5
c2 = 1.5
bounds = [-10, 10]

# 3. Khởi tạo
position = np.random.uniform(bounds[0], bounds[1], (num_particles, num_dimensions))
velocity = np.random.uniform(-1, 1, (num_particles, num_dimensions))
pbest_position = position.copy()
pbest_value = np.array([objective_function(p) for p in position])
gbest_index = np.argmin(pbest_value)
gbest_position = pbest_position[gbest_index].copy()
gbest_value = pbest_value[gbest_index]

# --- DANH SÁCH LƯU LỊCH SỬ HỘI TỤ ---
convergence_history = []  # <--- Biến này để lưu giá trị gBest qua từng vòng lặp

# 4. Vòng lặp
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

    # Lưu lại giá trị tốt nhất hiện tại vào lịch sử
    convergence_history.append(gbest_value)

    print(f"Iter {i + 1}: Best Value = {gbest_value:.6f}")

# 5. VẼ BIỂU ĐỒ HỘI TỤ (Trong PyCharm)
plt.plot(convergence_history, color='red', marker='o')
plt.title('Biểu đồ hội tụ PSO (PSO Convergence)')
plt.xlabel('Số vòng lặp (Iteration)')
plt.ylabel('Giá trị tốt nhất (Best Fitness)')
plt.grid(True)
plt.show()  # <--- Lệnh này sẽ bật cửa sổ biểu đồ lên
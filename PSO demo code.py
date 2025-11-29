import numpy as np
import matplotlib.pyplot as plt


# --- 1. Hàm mục tiêu (Sphere) ---
def objective_function(x):
    return np.sum(x ** 2)


# --- 2. Hàm thực thi PSO ---
def run_pso_algorithm(run_id, n_particles=30, n_dim=2, n_iter=100):
    # Tham số
    w = 0.7;
    c1 = 1.5;
    c2 = 1.5
    bounds = [-10, 10]

    # Khởi tạo
    position = np.random.uniform(bounds[0], bounds[1], (n_particles, n_dim))
    velocity = np.random.uniform(-1, 1, (n_particles, n_dim))

    pbest_pos = position.copy()
    pbest_val = np.array([objective_function(p) for p in position])

    gbest_index = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_index].copy()
    gbest_val = pbest_val[gbest_index]

    # List lưu lịch sử
    convergence_history = []

    for i in range(n_iter):
        r1 = np.random.rand(n_particles, n_dim)
        r2 = np.random.rand(n_particles, n_dim)

        # Cập nhật vận tốc & vị trí
        velocity = (w * velocity) + \
                   (c1 * r1 * (pbest_pos - position)) + \
                   (c2 * r2 * (gbest_pos - position))

        position = position + velocity
        position = np.clip(position, bounds[0], bounds[1])

        # Đánh giá
        for k in range(n_particles):
            fit = objective_function(position[k])

            if fit < pbest_val[k]:
                pbest_val[k] = fit
                pbest_pos[k] = position[k].copy()

                if fit < gbest_val:
                    gbest_val = fit
                    gbest_pos = position[k].copy()

        convergence_history.append(gbest_val)

    print(f"Run {run_id}: gBest cuối cùng = {gbest_val}")  # In giá trị thực tế
    return convergence_history


# --- 3. Chạy 5 lần với 100 vòng lặp ---
num_runs = 5
iterations = 100  # <--- ĐÃ CẬP NHẬT LÊN 100
all_histories = []

print(f"--- BẮT ĐẦU CHẠY {num_runs} LẦN (100 Iterations) ---")

for i in range(1, num_runs + 1):
    history = run_pso_algorithm(run_id=i, n_iter=iterations)
    all_histories.append(history)

# --- 4. Vẽ biểu đồ ---
plt.figure(figsize=(10, 6))
for i, history in enumerate(all_histories):
    plt.plot(history, label=f'Run {i + 1}')

plt.title(f'Hội tụ của PSO qua {iterations} vòng lặp')
plt.xlabel('Iterations')
plt.ylabel('gBest Value (Log Scale)')
plt.yscale('log')  # Log scale rất quan trọng khi số quá nhỏ
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.show()
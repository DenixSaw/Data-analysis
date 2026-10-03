from flyhash import FlyHash

BASE_WEIGHT_PATH = Path("./weights")
full_path = BASE_WEIGHT_PATH / "mnist_exact_k10.pkl"


with open(full_path, "rb") as f:
    gt_data = pickle.load(f)
exact_indices = np.asarray(gt_data["indices"])

K = K_NEIGHBORS
INPUT_DIM = X_train.shape[1]
HASH_DIM  = 7840
SPARSITY  = 0.05

fly = FlyHash(
    input_dim=INPUT_DIM,
    hash_dim=HASH_DIM,
    sparsity=SPARSITY,
    seed=SEED,
)


def hash_and_binarize(model, X):
    hs = model(X)
    hs = (hs > 0)
    if hasattr(hs, "toarray"):
        hs = hs.toarray()
    return hs.astype(np.int8)


t0 = time.perf_counter()
train_hashes = hash_and_binarize(fly, X_train)
time_hash_train_fly = time.perf_counter() - t0


t1 = time.perf_counter()
test_hashes = hash_and_binarize(fly, X_test)
time_hash_test_fly = time.perf_counter() - t1


nn_fly_hash = NearestNeighbors(
    n_neighbors=K,
    algorithm="brute",
    metric="hamming",
)
nn_fly_hash.fit(train_hashes)

tracemalloc.start()
start_time = time.perf_counter()

_, approx_indices_fly = nn_fly_hash.kneighbors(test_hashes)

time_search_fly = time.perf_counter() - start_time
_, peak_memory_fly = tracemalloc.get_traced_memory()
tracemalloc.stop()


# ---------- 6. Отчёт ----------
print("=== Результаты FlyHash ===")
print(f"Параметры: hash_dim={HASH_DIM}, sparsity={SPARSITY}")
print(f"Хеширование Train:         {time_hash_train_fly:.4f} сек")
print(f"Хеширование Test:          {time_hash_test_fly:.4f} сек")
print(f"Время поиска:              {time_search_fly:.4f} сек")
print(f"Пиковое потребление памяти: {peak_memory_fly / (1024 * 1024):.2f} MB (выделено Python)\n")

for k in (1, 5, 10):
    print(f"Recall@{k}: {calculate_recall_at_k(exact_indices, approx_indices_fly, k):.4f}")
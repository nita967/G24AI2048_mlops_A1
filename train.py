# train.py
from sklearn.tree import DecisionTreeRegressor
from misc import load_data, run_repeated_experiment, save_model, save_results_text
import os
import json

def main():
    df = load_data()
    model = DecisionTreeRegressor(random_state=0, max_depth=6)
    avg_mse, mses, trained_model = run_repeated_experiment(model, df, n_runs=5, test_size=0.2, scale=True)

    os.makedirs("artifacts", exist_ok=True)
    save_model(trained_model, "artifacts/dtree_model.joblib")
    result_text = f"DecisionTree Average MSE: {avg_mse}\nAll MSEs: {mses}\n"
    print(result_text)
    save_results_text("artifacts/dtree_results.txt", result_text)

    # also write a JSON for easier parsing in CI
    with open("artifacts/dtree_results.json", "w") as f:
        json.dump({"avg_mse": avg_mse, "mses": mses}, f)

if __name__ == "__main__":
    main()

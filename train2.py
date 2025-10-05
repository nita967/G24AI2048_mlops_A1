# train2.py
from sklearn.kernel_ridge import KernelRidge
from misc import load_data, run_repeated_experiment, save_model, save_results_text
import os
import json

def main():
    df = load_data()
    # KernelRidge — tune alpha and kernel as needed. Default here:
    model = KernelRidge(alpha=1.0, kernel='rbf')
    avg_mse, mses, trained_model = run_repeated_experiment(model, df, n_runs=5, test_size=0.2, scale=True)

    os.makedirs("artifacts", exist_ok=True)
    save_model(trained_model, "artifacts/kernelridge_model.joblib")
    result_text = f"KernelRidge Average MSE: {avg_mse}\nAll MSEs: {mses}\n"
    print(result_text)
    save_results_text("artifacts/kernelridge_results.txt", result_text)

    with open("artifacts/kernelridge_results.json", "w") as f:
        json.dump({"avg_mse": avg_mse, "mses": mses}, f)

if __name__ == "__main__":
    main()

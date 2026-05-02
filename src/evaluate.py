from src.config import ERROR_THRESHOLD

def evaluate_model(error):
    print("\n[Agent] Evaluating model performance...")

    if error > ERROR_THRESHOLD:
        print(f"[Agent Decision] Error {error:.2f} > {ERROR_THRESHOLD} → Retraining required")
        return False
    else:
        print(f"[Agent Decision] Error acceptable → Model approved")
        return True

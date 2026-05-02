from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_range

def run_pipeline():
    print("===== EV Range Agentic Pipeline =====")

    # Step 1: Train model
    error = train_model()

    # Step 2: Agent evaluates
    is_acceptable = evaluate_model(error)

    # Step 3: Autonomous decision
    if not is_acceptable:
        print("[Agent Action] Retraining model...")
        error = train_model()
        print("[Agent] Retraining complete")

    # Step 4: Prediction
    sample_input = [75, 50, 30, 220]
    result = predict_range(sample_input)

    print("\nFinal Prediction:")
    print(f"Input: {sample_input}")
    print(f"Predicted Range: {result:.2f} km")

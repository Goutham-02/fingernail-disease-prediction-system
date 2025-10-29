
from integrated_nail_diagnosis import NailDiagnosisSystem, convert_to_serializable
import json

# Initialize the system
system = NailDiagnosisSystem(
    model_path="17classes_resnet_97.pth",
    disease_data_path="data1.json"
)

# Example 1: Single image with symptoms
print("="*70)
print("EXAMPLE 1: Single image with symptoms")
print("="*70)

results = system.diagnose(
    image_paths=["dummy_dataset/1.PNG"],
    symptoms="patchy hair loss, itching"
)

print("\nTop prediction:", results['aggregated_prediction']['predicted_disease'])
print("Confidence:", f"{results['aggregated_prediction']['aggregated_score']*100:.1f}%")

# Example 2: Multiple images without symptoms
print("\n" + "="*70)
print("EXAMPLE 2: Multiple images without symptoms")
print("="*70)

results = system.diagnose(
    image_paths=[
        "dummy_dataset/10.PNG",
        "dummy_dataset/11.PNG",
        "dummy_dataset/12.PNG"
    ],
    symptoms="itching"
)

print("\nTop prediction:", results['aggregated_prediction']['predicted_disease'])
print("Confidence:", f"{results['aggregated_prediction']['aggregated_score']*100:.1f}%")

# Example 3: Access detailed results
print("\n" + "="*70)
print("EXAMPLE 3: Accessing detailed results")
print("="*70)

# Individual image predictions
for img_result in results['individual_predictions']:
    print(f"\nImage {img_result['image_id']}: {img_result['predicted_class']}")
    print(f"  Grad-CAM: {img_result['gradcam_path']}")
    print("  Top 3 predictions:")
    for pred in img_result['top_predictions']:
        print(f"    - {pred['disease']}: {pred['combined_score']*100:.1f}%")

# Aggregated prediction breakdown
print("\nAggregated Score Breakdown:")
breakdown = results['aggregated_prediction']['score_breakdown']
print(f"  Model Probability: {breakdown['model_probability']*100:.1f}%")
print(f"  Visual Evidence: {breakdown['visual_evidence']*100:.1f}%")
print(f"  Symptom Match: {breakdown['symptom_match']*100:.1f}%")

# Disease information
print("\nDisease Information:")
disease_info = results['disease_information']
print(f"  Category: {disease_info.get('category', 'N/A')}")
print(f"  Definition: {disease_info.get('definition', 'N/A')}")
print(f"  When to see doctor: {', '.join(disease_info.get('when_to_see_doctor', []))}")

# Save pretty-printed JSON
with open('diagnosis_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=convert_to_serializable)
print("\nFull results saved to: diagnosis_results.json")
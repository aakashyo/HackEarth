import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import os

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_emotion_distribution(labels, label_names, title, save_path):
    counts = np.bincount(labels, minlength=len(label_names))
    sorted_indices = np.argsort(counts)[::-1]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(label_names)), counts[sorted_indices], color='steelblue')
    plt.xticks(range(len(label_names)), [label_names[i] for i in sorted_indices], rotation=45, ha='right', fontsize=9)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_report(y_true, y_pred, labels, save_path):
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    return report


def main():
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    with open(os.path.join(MODEL_PATH, "label_info.json"), 'r', encoding='utf-8') as f:
        label_info = json.load(f)
    
    primary_labels = label_info['primary_labels']
    secondary_labels = label_info['secondary_labels']
    
    primary_preds = np.load(os.path.join(MODEL_PATH, "primary_preds.npy"))
    primary_true = np.load(os.path.join(MODEL_PATH, "primary_labels.npy"))
    secondary_preds = np.load(os.path.join(MODEL_PATH, "secondary_preds.npy"))
    secondary_true = np.load(os.path.join(MODEL_PATH, "secondary_labels.npy"))
    
    print("Generating Primary Emotion Confusion Matrix...")
    plot_confusion_matrix(
        primary_true, primary_preds, primary_labels,
        "Primary Emotion Confusion Matrix",
        os.path.join(OUTPUT_PATH, "primary_confusion_matrix.png")
    )
    
    print("Generating Secondary Emotion Confusion Matrix...")
    plot_confusion_matrix(
        secondary_true, secondary_preds, secondary_labels,
        "Secondary Emotion Confusion Matrix",
        os.path.join(OUTPUT_PATH, "secondary_confusion_matrix.png")
    )
    
    print("Generating Emotion Distribution Plots...")
    plot_emotion_distribution(
        primary_true, primary_labels,
        "Primary Emotion Distribution (Test Set)",
        os.path.join(OUTPUT_PATH, "primary_distribution.png")
    )
    
    plot_emotion_distribution(
        secondary_true, secondary_labels,
        "Secondary Emotion Distribution (Test Set)",
        os.path.join(OUTPUT_PATH, "secondary_distribution.png")
    )
    
    print("Generating Classification Reports...")
    primary_report = generate_report(
        primary_true, primary_preds, primary_labels,
        os.path.join(OUTPUT_PATH, "primary_classification_report.json")
    )
    
    secondary_report = generate_report(
        secondary_true, secondary_preds, secondary_labels,
        os.path.join(OUTPUT_PATH, "secondary_classification_report.json")
    )
    
    print("\n" + "="*50)
    print("PRIMARY EMOTION METRICS")
    print("="*50)
    print(f"Accuracy: {primary_report['accuracy']:.4f}")
    print(f"Weighted F1: {primary_report['weighted avg']['f1-score']:.4f}")
    print(f"Macro F1: {primary_report['macro avg']['f1-score']:.4f}")
    
    print("\n" + "="*50)
    print("SECONDARY EMOTION METRICS")
    print("="*50)
    print(f"Accuracy: {secondary_report['accuracy']:.4f}")
    print(f"Weighted F1: {secondary_report['weighted avg']['f1-score']:.4f}")
    print(f"Macro F1: {secondary_report['macro avg']['f1-score']:.4f}")
    
    print(f"\nVisualization saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

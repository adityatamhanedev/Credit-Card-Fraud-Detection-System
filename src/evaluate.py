from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(y_test, y_pred):
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    with open("outputs/report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
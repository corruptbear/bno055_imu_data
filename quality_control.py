import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder



def test_file(csvpath, fixed_len = 100):
    csvpath = os.path.abspath(os.path.expanduser(csvpath))
    df = pd.read_csv(csvpath)
    #df = pd.read_csv("./datasets_audio/raw_data/ble_imu_data_250516_180002_unit_converted_labeled.csv")

    # Filter out "moving" and "stop"
    df = df[~df["label"].isin(["moving", "stop"])].copy()

    # Create a segment ID by detecting label change
    df["segment_id"] = (df["label"] != df["label"].shift()).cumsum()

    # Now you can group by segment_id
    segments = []
    segment_labels = []
    for seg_id, seg_df in df.groupby("segment_id"):
        label = seg_df["label"].iloc[0]
        data = seg_df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].to_numpy()

        n_samples = data.shape[0]
        n_chunks = int(round(n_samples / fixed_len))
        print(n_samples)

        for i in range(n_chunks):
            chunk = data[i * fixed_len : (i + 1) * fixed_len]

            # Pad the last chunk if it's short
            if chunk.shape[0] < fixed_len:
                chunk = np.pad(chunk, ((0, fixed_len - chunk.shape[0]), (0, 0)), mode='edge')

            segments.append(chunk.flatten())
            segment_labels.append(label)


    X = np.array(segments)
    print(segment_labels)

    labels = np.array(segment_labels)
    #print(labels)

    """
    tsne = TSNE(n_components=2, perplexity=20, learning_rate='auto', init='pca', random_state=42)

    X_tsne = tsne.fit_transform(X)

    # Convert labels to colors
    palette = sns.color_palette("tab10", n_colors=len(set(labels)))
    label_to_color = {label: palette[i] for i, label in enumerate(sorted(set(labels)))}
    colors = [label_to_color[label] for label in labels]
    """

    """
    # 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=colors, s=60, edgecolor='k', alpha=0.9)

    ax.set_title("3D t-SNE of IMU Segments")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    plt.tight_layout()
    plt.show()
    """

    """
    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=labels, style=labels, palette='tab10',
        s=60, edgecolor='k', alpha=0.9
    )

    ax.set_title("t-SNE Visualization of IMU Segments")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Label")
    plt.tight_layout()
    plt.show()
    """


    """
    import umap.umap_ as umap

    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)

    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, palette='tab10')
    plt.title("UMAP Projection of IMU Segments")
    plt.show()
    """

    """
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X)

    df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3", "PC4"])
    df_plot["label"] = labels

    sns.pairplot(df_plot, hue="label", palette='tab10')
    plt.suptitle("PCA Pairplot of Segments", y=1.02)
    plt.show()
    """

    n_classes = len(set(labels))
    print(f"number of classes: {n_classes}")
    pred_clusters = KMeans(n_clusters=n_classes).fit_predict(X)

    print(pred_clusters)
    ari = adjusted_rand_score(labels, pred_clusters)
    print(f"Adjusted Rand Index: {ari:.3f}")

    """
    sil_vals = silhouette_samples(X, pred_clusters)
    plt.bar(range(len(sil_vals)), sil_vals)
    plt.axhline(np.mean(sil_vals), color="red", linestyle="--")
    plt.title("Silhouette Scores per Segment")
    plt.xlabel("Segment")
    plt.ylabel("Silhouette Score")
    plt.show()
    """


    # Desired label order
    label_order = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    
    le = LabelEncoder()
    le.classes_ = np.array(label_order)  # set classes manually
    y_encoded = le.transform(labels)     # encode labels in the correct order

    # 5-fold stratified cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True)

    all_y_true = []
    all_y_pred = []

    print("📊 5-Fold Cross-Validation (Random Forest)")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y_encoded)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        print("shape",X_train.shape)

        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        acc = accuracy_score(y_test, y_pred)
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")

    # Final evaluation across all folds
    print("\n✅ Overall Accuracy:", accuracy_score(all_y_true, all_y_pred))
    print("\n📋 Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=le.classes_, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Random Forest Confusion Matrix (5-Fold)")
    plt.tight_layout()
    plt.show()


# Load your data


#test_file("./datasets_audio/raw_data/ble_imu_data_250516_115403_unit_converted_labeled.csv",fixed_len = 104) #me
test_file("/Users/lws/Downloads/exp_data/p3/metronome_padded_bpm60_audio_imu_logs_250525_151630/ble_imu_data_250525_150916_unit_converted_labeled.csv", fixed_len=104) #kevin
#test_file("./datasets_audio/raw_data/ble_imu_data_250516_180002_unit_converted_labeled.csv",fixed_len = 104) #me
#test_file("./datasets_audio/raw_data/ble_imu_data_250516_111202_unit_converted_labeled.csv",fixed_len = 78) #me bpm 80

#test_file("./datasets_audio/raw_data/ble_imu_data_250520_161721_unit_converted_labeled.csv",fixed_len = 104) #gabe
#test_file("./datasets_audio/raw_data/ble_imu_data_250523_142720_unit_converted_labeled.csv",fixed_len = 104) #ehsan1
#test_file("./datasets_audio/raw_data/ble_imu_data_250523_160722_unit_converted_labeled.csv",fixed_len = 89) #ehsan2 bpm 70


#test_file("./datasets_audio/raw_data/ble_imu_data_250520_161721_unit_converted_labeled.csv",fixed_len = 104) #pat
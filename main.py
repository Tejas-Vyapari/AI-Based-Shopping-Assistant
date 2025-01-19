import os
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from collections import Counter
import matplotlib.pyplot as plt
import cv2
from fuzzywuzzy import fuzz, process  # For fuzzy matching

# Global variable to store user search history
search_history = []

# def search_images(query):

def load_images(keyword):
    """
    Load and allow navigation through images of a given category from the 'data' directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: The directory '{data_path}' does not exist.")

    for category in os.listdir(data_path):
        if category.lower() == keyword.lower():
            category_path = os.path.join(data_path, category)
            image_files = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
            if not image_files:
                print(f"No images found for category: {keyword}")
                return

            print(f"Displaying images for category: {keyword}")
            index = 0
            while True:
                img = cv2.imread(image_files[index])
                if img is not None:
                    img_resized = cv2.resize(img, (400, 400))
                    cv2.imshow(f"Category: {category} (Use ←/→ to navigate, ESC to exit)", img_resized)

                key = cv2.waitKey(0) & 0xFF
                if key == 27:  # ESC key
                    cv2.destroyAllWindows()
                    return
                elif key in [ord('a'), 81]:  # Left arrow key or 'a'
                    index = (index - 1) % len(image_files)
                elif key in [ord('d'), 83]:  # Right arrow key or 'd'
                    index = (index + 1) % len(image_files)

            cv2.destroyAllWindows()
            return

    print(f"No category found for: {keyword}")


def find_largest_cluster(label_counts):
    """Find the label of the largest cluster."""
    return max(label_counts, key=label_counts.get)


def recommend_items(X, y, labels, cluster_centers, name_list, keyword):
    """
    Generate and display recommendations based on search history or clustering.
    """
    global search_history

    # Fuzzy matching to find relevant products based on the search term
    matched_products = process.extract(keyword, name_list, limit=10, scorer=fuzz.partial_ratio)
    search_based_recommendations = [idx for idx, score in enumerate(matched_products) if score[1] > 70]

    if search_based_recommendations:
        print("\nRecommendations based on your search history:")
        for idx in search_based_recommendations:
            product_idx = name_list.index(matched_products[idx][0])
            print(f"Item ID: {y[product_idx][0]}   Item Name: {name_list[product_idx]}")
        return

    # Fall back to cluster-based recommendations
    print("\nNo recommendations based on search history. Showing cluster-based recommendations.")
    label_counts = Counter(labels)
    max_label = find_largest_cluster(label_counts)
    suggest_indices = [i for i, label in enumerate(labels) if label == max_label]

    suggested_items = [y[i][0] for i in suggest_indices]
    suggested_names = [name_list[i] for i in suggest_indices]

    for item_id, item_name in zip(suggested_items[:8], suggested_names[:8]):
        print(f"Item ID: {item_id}   Item Name: {item_name}")

    # Visualize clusters
    colors = ['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']
    while len(colors) < len(set(labels)):
        colors.extend(colors)

    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i] % len(colors)], markersize=10)

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker="x", s=150, c="black", linewidths=3, zorder=10)
    plt.title("Cluster Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def main():
    """
    Main function to handle user input, image loading, and product recommendation.
    """
    global search_history

    while True:
        keyword = input("Search for a product category (or type 'exit' to quit): ").strip()
        if keyword.lower() == 'exit':
            print("Exiting the application.")
            break

        # Add keyword to search history
        search_history.append(keyword)
        load_images(keyword)

        # Dataset path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'data', 'person.csv')

        if not os.path.exists(csv_path):
            print(f"Error: 'person.csv' file not found in the 'data' directory at {csv_path}.")
            return

        # Load the dataset
        try:
            dataset = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print("Error: 'person.csv' file is empty.")
            return

        # Extract features and item details
        X = dataset.iloc[:, [2, 3]].values  # Features
        y = dataset.iloc[:, [0]].values  # Item IDs
        name_list = dataset['Item_names'].tolist()

        # Perform clustering
        ms = MeanShift()
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        # Generate recommendations
        recommend_items(X, y, labels, cluster_centers, name_list, keyword)


if __name__ == "__main__":
    main()

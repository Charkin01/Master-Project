# Apply sanitization techniques
def sanitize_dataset(tokenized_dataset):
    def apply_isolation_forest(data):
        iso_forest = IsolationForest(contamination=0.1)
        preds = iso_forest.fit_predict(data)
        return data[preds == 1]

    # Extract features and retain the structure of the dataset
    features = [example['input_ids'] for example in tokenized_dataset]
    features = np.array(features)
    sanitized_features = apply_isolation_forest(features)

    def apply_kmeans(data):
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(data)
        return data[kmeans.labels_ == 0]

    clustered_features = apply_kmeans(sanitized_features)
    unique_features = np.unique(clustered_features, axis=0)

    sanitized_dataset = []
    for feature in unique_features:
        idx = int(np.where((features == feature).all(axis=1))[0][0])  # Convert numpy integer to Python integer
        sanitized_dataset.append(tokenized_dataset[idx])

    return sanitized_dataset

sanitized_tokenized_dataset = sanitize_dataset(tokenized_dataset)
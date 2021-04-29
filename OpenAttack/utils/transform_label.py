
def update_label(dataset, labels_to_labels):
    """
    :param datasets dataset: The huggingface datasets you use.
    :param dict labels_to_labels: map the origin labels to the labels you want.

    :Package Requirements:
        * **datasets**

    """
    features = [ kw for kw in dataset.features ]
    for kw in features:
        if kw not in labels_to_labels:
            dataset = dataset.remove_columns([kw])
    for key, value in labels_to_labels.items():
        dataset = dataset.rename_column(key, value)
    return dataset
    
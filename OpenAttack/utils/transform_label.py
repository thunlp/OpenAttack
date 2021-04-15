
def update_label(dataset, labels_to_labels):
    """
    :param datasets dataset: The huggingface datasets you use.
    :param dict labels_to_labels: map the origin labels to the labels you want.

    :Package Requirements:
        * **datasets**

    """
    import datasets
    for key, value in labels_to_labels.items():
        dataset.rename_column_(key, value)
    return dataset
    
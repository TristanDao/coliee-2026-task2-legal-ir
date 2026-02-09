from data import Task2Dataset

data = Task2Dataset(
    data_dir = "Data/task2_train_files_2025",
    label_path = "Data/task2_train_labels_2025.json"
)

print(data[0])
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

class Preprocessor:
    def __init__(self, file_path, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_labels(self, path=None, labels=[]):
        if labels:
            self.labels = labels
        elif path:
            try:
                self.labels = pd.read_csv(path).iloc[:, 0].tolist()
            except Exception as e:
                raise ValueError(f"Cannot load label in {path}: {e}")
        else:
            raise ValueError(f"No label provided. Please, prove an array or an path for a CSV containing the labels")

    def process(self):
        if not self.labels:
            raise ValueError("Labels not loaded, please load labels with method 'load_labels' and providing an label array or a path for CSV file.")
        
        df = pd.read_csv(self.file_path)
        if len(df.columns) != len(self.labels):
            raise ValueError("The number of columns does not match with the provided labels")
        df.columns = self.labels
        
        df['label_encoded'] = self.label_encoder.fit_transform(df['label'])
        X = df.drop(columns=['label', 'label_encoded'])
        y = df['label_encoded']
        X = self.scaler.fit_transform(X.select_dtypes(include=['float64', 'int64']))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, 
                                                            stratify=y, random_state=self.random_state)
        return X_train, X_test, y_train, y_test
    
    def process_another_datasets(self, file_paths):
        if not self.labels:
            raise ValueError("The labels need to be proccesed before adding another datasets.")
        datasets = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            if len(df.columns) != len(self.labels):
                raise ValueError(f"The number of columns in the file {file_path} does not match with the provided labels.")
            df.columns = self.labels

            df['label_encoded'] = self.label_encoder.transform(df['label'])
            X = df.drop(columns=['label', 'label_encoded'])
            y = df['label_encoded']
            X = self.scaler.transform(X.select_dtypes(include=['float64', 'int64']))
            datasets.append((torch.tensor(X, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)))
        
        return datasets
        
    @staticmethod
    def add_noise(X, noise_type='gaussian', noise_level=0.01, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)

        if noise_type == 'gaussian':
            noise = np.random.normal(loc=0, scale=noise_level, size=X.shape)
            X_noisy = X + noise

        elif noise_type == 'uniform':
            noise = np.random.uniform(low=-noise_level, high=noise_level, size=X.shape)
            X_noisy = X + noise

        elif noise_type == 'salt_pepper':
            X_noisy = X.copy()
            n_samples, n_features = X.shape
            n_salt = int(noise_level * X.size * 0.5)
            n_pepper = int(noise_level * X.size * 0.5)

            coords = [np.random.randint(0, i - 1, n_salt) for i in X.shape]
            X_noisy[coords] = X.max()

            coords = [np.random.randint(0, i - 1, n_pepper) for i in X.shape]
            X_noisy[coords] = X.min()

        else:
            raise ValueError(f"Unsupported noise type '{noise_type}'")

        return X_noisy

    def process_with_noise(self, noise_type='gaussian', noise_level=0.01):
        if not self.labels:
            raise ValueError("Labels not loaded. Use 'load_labels' first.")
        
        df = pd.read_csv(self.file_path)
        if len(df.columns) != len(self.labels):
            raise ValueError("The number of columns does not match the provided labels.")
        df.columns = self.labels
        
        df['label_encoded'] = self.label_encoder.fit_transform(df['label'])
        X = df.drop(columns=['label', 'label_encoded']).values
        y = df['label_encoded'].values
        
        X = self.scaler.fit_transform(X)

        X_noisy = self.add_noise(X, noise_type=noise_type, noise_level=noise_level)

        return X_noisy, y

    def generate_synthetic_dataset(self, noise_type='gaussian', noise_level=0.01):
        X_noisy, y = self.process_with_noise(noise_type=noise_type, noise_level=noise_level)
        df_noisy = pd.DataFrame(X_noisy, columns=self.labels[:-2])
        df_noisy['label'] = self.label_encoder.inverse_transform(y)
        
        return df_noisy
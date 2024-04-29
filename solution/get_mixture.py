# =============================================================================
# 以下是一个采用随机采样的示例，参赛者可自由编写文件内容。
# =============================================================================

import json
import random
from pathlib import Path
import os
import gc
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.decomposition import PCA
from tqdm import tqdm
import random
import torch
import numpy as np
from datasets import load_dataset

# 设置随机种子，保证结果可复现

random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# 如果使用 NumPy
# import numpy as np
# np.random.seed(seed)

# 如果使用 PyTorch
# import torch
# torch.manual_seed(seed)

# 开发套件根目录
base_dir = Path(__file__).resolve().parent

# 输入输出路径
input_dir = base_dir / "input"
ratio_path = base_dir / "output" / "sft_data" / "ratio.json"
mixture_path = base_dir / "output" / "sft_data" / "mixture.jsonl"

file_list = list(input_dir.glob("*.jsonl"))
file_names = [file.name for file in file_list]
_file_names = [name.replace(".jsonl", "") for name in file_names]

model_checkpoint = "distilbert-base-uncased"
total_sample_count = 50000


class DistributionEstimator:
    def __init__(self, vectors: np.ndarray, name="unknown_name"):
        self.vectors = vectors
        assert vectors.ndim == 2
        self.n = vectors.shape[0]
        self.vector_dim = vectors.shape[1]
        self.L = np.linalg.cholesky(self.__get_cov())
        self.mean = self.__get_mean()
        self.name = name

    def __get_mean(self):
        return self.vectors.mean(axis=0)

    def __get_cov(self):
        return np.cov(self.vectors, rowvar=False)

    def __get_single_sample(self):
        z = np.random.randn(self.vector_dim)
        x = self.mean + self.L @ z

        return x

    def __find_nearest_original(self, x):
        manhattan_distances = np.abs(self.vectors - x).sum(axis=1)
        min_idx = np.argmin(manhattan_distances)

        return min_idx

    # id bi excel xiao 2
    def get_sample_id_sets(self, count):
        id_set = set()
        pbar = tqdm(total=count, desc='Generating sample IDs of ' + str(self.name))
        while len(id_set) < count:
            x = self.__get_single_sample()
            original_id = self.__find_nearest_original(x)
            id_set.add(original_id)
            pbar.update(1)
        pbar.close()
        return id_set


def generate_mixture(input_dir, ratio_path, mixture_path):
    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_checkpoint,trust_remote_code=True)
    tokenizer = DistilBertTokenizer.from_pretrained(model_checkpoint)
    model = DistilBertModel.from_pretrained(model_checkpoint)
    device = torch.device("cuda")
    model.to(device)

    file_name_to_word_vectors_dict = {}

    def get_TinyBERT_embeddings(input_ids, attention_mask):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    def get_embeddings(data, batch_size=32):
        embeddings = []
        for i in tqdm(range(0, len(data['text']), batch_size), desc="Generating embeddings"):
            batch_texts = data['text'][i:i + batch_size].tolist()

            encoding = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256,
                                 return_attention_mask=True)
            batch_input_ids = encoding.input_ids
            attention_mask = encoding.attention_mask

            batch_embeddings = get_TinyBERT_embeddings(batch_input_ids, attention_mask)
            embeddings.extend(batch_embeddings)

        return np.vstack(embeddings)

    for num, file_name in enumerate(file_list):
        dataset = load_dataset('json', data_files=str(file_name))
        label = dataset['train']['text']
        df_output = pd.DataFrame(label, columns=['text'])
        train_embeddings = get_embeddings(df_output, batch_size=32)
        pca = PCA(n_components=100)
        reduced_embeddings = pca.fit_transform(train_embeddings)
        file_name_to_word_vectors_dict["{}".format(_file_names[num])] = reduced_embeddings
        gc.collect()
        print("{}  ".format(str(file_list[num])))

    total_original_count = 0
    csv_path_to_original_count = dict()
    for path in _file_names:
        count = file_name_to_word_vectors_dict[path].shape[0]
        csv_path_to_original_count[path] = count
        total_original_count += count

    csv_path_to_sample_count = dict()
    for path in _file_names:
        sample_count = csv_path_to_original_count[path] / total_original_count * total_sample_count
        csv_path_to_sample_count[path] = round(sample_count)

    def sample_from_chosen_dataset(data_dict, file_name_key, sample_count_the_file):
        ndarray_data = data_dict[file_name_key]
        distribution_estimator = DistributionEstimator(ndarray_data, name=file_name_key)
        id_set = distribution_estimator.get_sample_id_sets(sample_count_the_file)
        return id_set

    id_sets = []
    for path, sample_count in tqdm(csv_path_to_sample_count.items(), total=len(csv_path_to_sample_count),
                                   desc='creating sample id set'):
        id_sets.append(sample_from_chosen_dataset(file_name_to_word_vectors_dict,path, sample_count))

    samples = []
    data = set()
    for num, id_set in enumerate(id_sets):
        lines = []
        # open json file
        with open(file_list[num], 'r') as file:
            for line_id in file:
                js = json.loads(line_id)
                lines.append(js)

        for line_id in id_set:
            samples.append(lines[line_id])

    with open(mixture_path, 'w') as f:
        for item in samples:
            json.dump(item, f)
            f.write('\n')

    print(mixture_path)

    def calculate_set_ratios(id_sets, total_count):
        set_ratios = []
        for id_set in id_sets:
            set_count = len(id_set)
            set_ratio = set_count / total_count
            set_ratios.append(set_ratio)
        return set_ratios

    total_count = sum(len(id_set) for id_set in id_sets)
    set_ratios = calculate_set_ratios(id_sets, total_count)

    ratio = {name: float(prob) for name, prob in zip(file_names, set_ratios)}

    with open(ratio_path, "w") as ratio_file:
        json.dump(ratio, ratio_file)

    # 执行函数


generate_mixture(input_dir, ratio_path, mixture_path)

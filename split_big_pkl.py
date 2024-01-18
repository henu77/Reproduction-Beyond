import pickle

def load_large_pickle(file_path, chunk_size=1000):
    with open(file_path, 'rb') as f:
        while True:
            try:
                data_chunk = []
                for _ in range(chunk_size):
                    # 逐行读取数据块
                    data_chunk.append(pickle.load(f))
            except EOFError:
                break

            yield data_chunk


import pickle
import os

def split_pickle_file(input_file, output_folder, chunk_size):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, 'rb') as f:
        try:
            while True:
                # 从原始文件中加载一部分数据
                data_chunk = pickle.load(f)

                # 生成新的输出文件名
                output_file = os.path.join(output_folder, f'chunk_{len(os.listdir(output_folder)) + 1}.pkl')

                # 将数据写入新的输出文件
                with open(output_file, 'wb') as out_file:
                    pickle.dump(data_chunk, out_file)

        except EOFError:
            pass



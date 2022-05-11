import os
import os.path as osp

from itertools import product

storage_dir = "./storage/"
pretrain_dataset_path = "./storage/data/"
pretrain_dataset_cache_path = "./storage/cached_mri/"
adni_dataset_path = "./storage/adni_data/"
adni_record_path = "./storage/ADNI1_Annual_2_Yr_3T_4_23_2022.csv"
max_epochs = 10

for seed in range(5):
    for hidden_size in [60]:
        cnn_checkpoint = osp.join(storage_dir, f"cnn_checkpoint_{hidden_size}_{seed}")
        os.makedirs(cnn_checkpoint, exist_ok=True)
        pretrain_transformer_dataset_cache = osp.join(storage_dir, f"transformer_dataset_cache_{hidden_size}_{seed}") 
        transformer_checkpoint = osp.join(storage_dir, f"transformer_checkpoint_{hidden_size}_{seed}")
        os.makedirs(transformer_checkpoint, exist_ok=True)

        # First generate CNN's that are not pretrained
        os.system(
           f"""python ./CNN.py --pretrain 0 --hidden_size {hidden_size} --checkpoint_path {cnn_checkpoint}/unpretrained"""
           f""" --raw_dataset {pretrain_dataset_path} --dataset_cache {pretrain_dataset_cache_path}""")

        # Then generate transformers that are not pretrained
        os.system( 
           f"""python ./pretrain_transformer.py --n_hidden {hidden_size} --pretrain 0 --store_checkpoint_path {transformer_checkpoint}/unpretrained"""
           f""" --imaging_dataset_dir {pretrain_dataset_path} --imaging_dataset_cache_dir {pretrain_dataset_cache_path} --cache_root {pretrain_transformer_dataset_cache}""")

        # Start actual training
        # Pretrain CNN
        os.system(
           f"""python ./CNN.py --hidden_size {hidden_size} --checkpoint_path {cnn_checkpoint}/pretrained --epochs {max_epochs}"""
           f""" --raw_dataset {pretrain_dataset_path} --dataset_cache {pretrain_dataset_cache_path}""")
        print("Finished pretraining cnn")

        # Pretrain transformer
        os.system(
           f"""python ./pretrain_transformer.py --store_checkpoint_path {transformer_checkpoint}/pretrained --max_epochs {max_epochs}"""
           f""" --imaging_dataset_dir {pretrain_dataset_path} --imaging_dataset_cache_dir {pretrain_dataset_cache_path} --cache_root {pretrain_transformer_dataset_cache}"""
           f""" --n_hidden {hidden_size} --cnn_checkpoint_path {cnn_checkpoint}/pretrained""")
        print("Finished pretraining transformer")

        # Whether classification or regression
        for task in [0, 1]:
            # Run without pretrained model
            print("Without pretrained")
            os.system(
                f"""python ./finetune.py --classification {task} --random_seed {seed} --n_hidden {hidden_size} --max_epochs {max_epochs}"""
                f""" --transformer_checkpoint_path {transformer_checkpoint}/unpretrained --cnn_checkpoint_path {cnn_checkpoint}/unpretrained"""
                f""" --csv_file_loc {adni_record_path} --adni_dataset {adni_dataset_path}""")
            print("With pretrained")
            for train_ratio in [1, 0.2, 0.5]:
                for tune_cnn, tune_transformer in product([0, 1], [0, 1]):
                    os.system(
                        f"""python ./finetune.py --pretrained 1 --classification {task} --finetune_cnn {tune_cnn} --finetune_transformer {tune_transformer}"""
                        f""" --random_seed {seed} --n_hidden {hidden_size} --train_ratio {train_ratio} --max_epochs {max_epochs}"""
                        f""" --transformer_checkpoint_path {transformer_checkpoint}/pretrained --cnn_checkpoint_path {cnn_checkpoint}/pretrained"""
                        f""" --csv_file_loc {adni_record_path} --adni_dataset {adni_dataset_path}""")


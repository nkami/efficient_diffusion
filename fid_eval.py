import matplotlib.pyplot as plt
from cleanfid import fid
import subprocess
import sys

if __name__ == '__main__':
    source_dataset = 'flowers'
    if source_dataset == 'ffhq':
        source_path = f'./data/128x128_{source_dataset}'
    else:
        source_path = f'./data/128x128_{source_dataset}/class_folder/128x128_{source_dataset}'
    models = [f'model_flower_p2gamma0._under_gamma0.5_2res_fp16_b256', 'model_flower_p2gamma0.5_uni_gamma0.5_2res_fp16_b256']
    legend_names = ['under05', 'uni05']
    # checkpoints = ['020000', '040000', '060000', '080000', '100000', '120000', '140000', '160000', '180000', '200000']
    checkpoints = ['010000', '020000', '030000', '040000', '050000', '060000', '070000', '080000', '090000', '100000']
    models_scores = []
    steps = [i for i in range(1, 11)]
    for model in models:
        cur_scores = []
        for ckpt in checkpoints:
            gen_path = f'./samples/{model}_{ckpt}'
            # completed_process = subprocess.run(['python', '-m', 'pytorch_fid', source_path, gen_path],
            #                                    capture_output=True, encoding="utf-8")
            # cur_fid = completed_process.stdout.split()[1]
            cur_score = fid.compute_fid(source_path, gen_path)
            cur_scores.append(float(cur_score))
            print(f'calculated score for {model} {ckpt}')
            sys.stdout.flush()
        models_scores.append(cur_scores)
    for cur_model_scores in models_scores:
        plt.plot(steps, cur_model_scores)
    plt.legend(legend_names)
    plt.title(f'{source_dataset} results')
    plt.xlabel(r'10k Training Steps')
    plt.ylabel('FID')
    plt.savefig(f'{source_dataset}.png')
    print('done')
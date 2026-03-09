import numpy as np
import pandas as pd
import os


def generate_sequence_data(
    csv_path: str,
    data_root: str,
    output_suffix: str,
    in_len: int,
    out_len: int,
    training_set_pct: float = 0.8,
) -> None:
    """Build sequence DataFrames from CSV and save train/test splits."""
    all_data = pd.read_csv(csv_path)
    # Modify paths for relevant columns
    # for col in ['unit1_rgb', 'unit1_radar', 'unit1_pwr_60ghz']:
    #     all_data[col] = all_data[col].apply(lambda x: x.replace('./unit1/', './dataset/scenario9/unit1/'))

    all_seq_idx = all_data['seq_index'].unique()
    all_seq_split = []

    for i in all_seq_idx:
        tmp = all_data[all_data['seq_index'] == i]  # belong to the same time stamp
        tmp = tmp[['unit1_rgb', 'unit1_radar', 'unit1_pwr_60ghz', 'seq_index']]
        # tmp = tmp[['unit1_rgb', 'unit1_pwr_60ghz', 'seq_index']]
        all_seq_split.append(tmp)

    all_seqs = []
    for seq in all_seq_split:  # iterate over each sequence, i.e., a DataFrame
        start = 0
        while start + in_len + out_len < seq.shape[0]:
            image = seq['unit1_rgb'][start:start + in_len].tolist()
            radar = seq['unit1_radar'][start:start + in_len].tolist()
            in_beam = seq['unit1_pwr_60ghz'][start:start + in_len].tolist()
            out_beam = seq['unit1_pwr_60ghz'][start + in_len:start + in_len + out_len].tolist()
            seq_idx = seq['seq_index'][0:1].tolist()  # first row of seq_index column
            all_seqs.append(image + radar + in_beam + out_beam + seq_idx)
            # all_seqs.append(image+in_beam+out_beam+seq_idx)
            start += 1

    col_names = [f'camera{i}' for i in range(1, in_len + 1)] + \
                [f'radar{i}' for i in range(1, in_len + 1)] + \
                [f'beam{i}' for i in range(1, in_len + 1)] + \
                [f'future_beam{i}' for i in range(1, out_len + 1)] + \
                ['seq_index']

    all_seqs = pd.DataFrame(all_seqs, columns=col_names)

    ind_select = int(training_set_pct * all_seq_idx.shape[0])
    train_seq_idx = np.sort(all_seq_idx[:ind_select])
    test_seq_idx = np.sort(all_seq_idx[ind_select:])

    train_seqs = all_seqs[all_seqs['seq_index'].isin(train_seq_idx)]
    test_seqs = all_seqs[all_seqs['seq_index'].isin(test_seq_idx)]

    train_seqs.to_csv(os.path.join(data_root, f'train_seqs{output_suffix}.csv'), index=False)
    test_seqs.to_csv(os.path.join(data_root, f'test_seqs{output_suffix}.csv'), index=False)


    print('file saved to: ', os.path.join(data_root, f'train_seqs{output_suffix}.csv'))
    print('file saved to: ', os.path.join(data_root, f'test_seqs{output_suffix}.csv'))

if __name__ == '__main__':
    # Configuration - modify these paths as needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    # Dataset paths
    data_root = os.path.join(current_dir, 'dataset', 'scenario9')


    in_len = 8
    out_len = 6
    training_set_pct = 0.8
    OUTPUT_SUFFIX = '_RA'
    csv_path = os.path.join(data_root, f'scenario9{OUTPUT_SUFFIX}.csv')

    generate_sequence_data(
        csv_path=csv_path,
        data_root=data_root,
        output_suffix=OUTPUT_SUFFIX,
        in_len=in_len,
        out_len=out_len,
        training_set_pct=training_set_pct,
    )
    print('done for output suffix: ', OUTPUT_SUFFIX)
  

    OUTPUT_SUFFIX = '_DA'
    csv_path = os.path.join(data_root, f'scenario9{OUTPUT_SUFFIX}.csv')
    generate_sequence_data(
        csv_path=csv_path,
        data_root=data_root,
        output_suffix=OUTPUT_SUFFIX,
        in_len=in_len,
        out_len=out_len,
        training_set_pct=training_set_pct,
    )
    print('done for output suffix: ', OUTPUT_SUFFIX)
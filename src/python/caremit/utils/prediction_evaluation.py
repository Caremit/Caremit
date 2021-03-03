import numpy as np
import pandas as pd


BEAT_MAP = {
    0: 'Normal beat (N)',
    1: 'Premature or ectopic supraventricular beat (S)',
    2: 'Premature ventricular contraction (V)',
    3: 'Fusion of ventricular and normal beat (F)',
    4: 'Unclassifiable beat (Q)'
}


# move to separate module eventually
def get_prediction_df(confidence_levels: np.array,
                      true_labels: np.array) -> pd.DataFrame:
    """
    Creates a pandas DataFrame from model output as basis for further processing
    Args:
        confidence_levels: model output
        true_labels: control data
    """
    df = pd.DataFrame(confidence_levels)
    df['predicted_label'] = df.idxmax(axis=1)
    df['true_label'] = true_labels
    df['prediction_correct'] = df['predicted_label'] == df['true_label']
    df['confidence'] = df \
        .loc[:, list(range(5))] \
        .max(axis=1)
    return df


def eval_prediction(df, print_all_wrong=False):
    """Helper to inspect model prediction quality"""
    incorrect_df = df.loc[~df['prediction_correct']].copy()
    print(f'Overall correctness: {(1 - len(incorrect_df) / len(df)) * 100:5.2f} %')

    print('\nCorrectness per category:')

    def fm(x):
        return pd.Series(data=len(x.loc[x['prediction_correct']]) / len(x),
                         index=['corr %'])

    res = df.groupby('true_label').apply(fm)
    print(res)

    print('\nHighest confidence for wrong predictions per category:')
    res = incorrect_df \
        .loc[:, ['predicted_label', 'true_label', 'confidence']] \
        .groupby(['predicted_label', 'true_label']) \
        .max()
    print(res)

    # show all confidence distribution for wrong predictions
    if print_all_wrong:
        print('\nConfidence distribution for wrong predictions:')
        sorted_df = incorrect_df \
            .sort_values(by='confidence', ascending=False)
        sorted_df.reset_index(inplace=True, drop=True)

        for row in sorted_df.itertuples():
            s = f'{str(row[0]).rjust(4)} '
            s += ', '.join([f'{row[idx]:5.9f}' for idx in range(1, 6)])
            s += f', {row.predicted_label}, {row.true_label}, {row.max_confidence:5.9f}'
            print(s)


def get_representative_signal_df(model_output: np.array) -> pd.DataFrame:
    """Extracts the signal with the highest confidence per classified category
    in the model output, and returns them as a Dataframe."""
    df = pd.DataFrame(model_output)
    df['predicted_id'] = df.idxmax(axis=1)
    df['predicted_label'] = df['predicted_id'].map(BEAT_MAP)
    df['confidence'] = df.loc[:, list(range(5))].max(axis=1)
    df.loc[:, ['predicted_label', 'confidence']].copy()

    def reduce_to_category(df_group):
        """Reduce the sub_dataframe (groupby object) to the entry with the
        maximum confidence"""
        max_idx = df_group['confidence'].idxmax()
        series = df_group.loc[max_idx, :]
        series['max_idx'] = max_idx
        series['category_size'] = len(df_group)
        return series

    return df \
        .groupby('predicted_label') \
        .apply(reduce_to_category)

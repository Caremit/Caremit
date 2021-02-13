from functools import wraps

import pandas as pd
import wfdb

@wraps(wfdb.rdsamp)
def get_df_from_sample(*args, **kwargs) -> pd.DataFrame:

    np_array, fields = wfdb.rdsamp(*args, **kwargs)
    cols = [fields['sig_name'][idx] + ' [' + fields['units'][idx] + ']'
            for idx in range(len(fields['sig_name']))]
    df = pd.DataFrame(data=np_array,
                      columns=cols)
    return df

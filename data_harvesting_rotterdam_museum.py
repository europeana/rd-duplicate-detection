import fire
import pyeuropeana.apis as apis
import pyeuropeana.utils as utils
from pathlib import Path

def main(**kwargs):
    saving_path = kwargs.get('saving_path')
    #dataset_name = kwargs.get('dataset_name')
    n_objects = kwargs.get('n_objects',12)

    #objects = apis.search(query = f'edm_datasetName:"{dataset_name}"', rows = n_objects)

    objects = apis.search(
        query = f'ster AND DATA_PROVIDER:"Museum Rotterdam"',
        rows = n_objects
    )

    df = utils.search2df(objects)
    df.to_csv(saving_path,index = False)
    print('Finished')


if __name__ == '__main__':
    fire.Fire(main)
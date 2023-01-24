import fire
from pathlib import Path
import pandas as pd
import pyeuropeana.utils as utils

def download_images(df,saving_dir,n_images = 10):

  print_every = 100

  df = df.sample(frac = 1).reset_index(drop=True)

  for i,row in df.iterrows():
    if i > n_images:
      break
    try:
      img = utils.url2img(row['image_url'])
    except:
      img = None
    if not img:
      continue

    fname = row['europeana_id'].replace('/','11ph11')+'.jpg'

    img.save(saving_dir.joinpath(fname))

    if i % print_every == 0:
      print(i)


def main(**kwargs):
    saving_dir = kwargs.get('saving_path')
    input = kwargs.get('csv_path')

    saving_dir = Path(saving_dir)

    saving_dir.mkdir(parents=True,exist_ok=True)

    df = pd.read_csv(input)
    print(f'Downloading {df.shape[0]} images')
    download_images(df,saving_dir,n_images = 1e6)
    print('Finished')

if __name__ == '__main__':
    fire.Fire(main)
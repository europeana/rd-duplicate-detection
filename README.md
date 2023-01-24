
# Setup

```
python3.8 -m venv img_dup
source img_dup/bin/activate
python3.8 -m pip install -U pip
python3.8 -m pip install fastdup
python3.8 -m pip install pyeuropeana
python3.8 -m pip install fire
```



# 2059516_EU_FD_Collections_Trust

## Data Harvesting

`nohup python3.8 data_harvesting.py --saving_path /home/jcejudo/projects/duplicate_detection/data/2059516_EU_FD_Collections_Trust.csv --dataset_name 2059516_EU_FD_Collections_Trust --n_objects 100000 &> data_harvesting.out &`

`nohup python3.8 download_images.py --saving_path /home/jcejudo/projects/duplicate_detection/data/2059516_EU_FD_Collections_Trust --csv_path /home/jcejudo/projects/duplicate_detection/data/2059516_EU_FD_Collections_Trust.csv &> download_images.out &`


## Apply duplicate detection

`nohup python3.8  run_duplicate_detection.py --input_dir /home/jcejudo/projects/duplicate_detection/data/2059516_EU_FD_Collections_Trust --work_dir /home/jcejudo/projects/duplicate_detection/results/2059516_EU_FD_Collections_Trust --num_images 500 --num_components 50 &> duplicate_detection.out &`



# Rotterdam Museum

## Data Harvesting

`nohup python3.8 data_harvesting_rotterdam_museum.py --saving_path /home/jcejudo/projects/duplicate_detection/data/rotterdam_museum.csv --n_objects 100000 &> data_harvesting.out &`

`nohup python3.8 download_images.py --saving_path /home/jcejudo/projects/duplicate_detection/data/rotterdam_museum --csv_path /home/jcejudo/projects/duplicate_detection/data/rotterdam_museum.csv &> download_images.out &`

## Apply duplicate detection

`nohup python3.8  run_duplicate_detection.py --input_dir /home/jcejudo/projects/duplicate_detection/data/rotterdam_museum --work_dir /home/jcejudo/projects/duplicate_detection/results/rotterdam_museum --num_images 500 --num_components 50 &> duplicate_detection.out &`



# 09102_Ag_EU_MIMO

## Data Harvesting

`nohup python3.8 data_harvesting.py --saving_path /home/jcejudo/projects/duplicate_detection/data/09102_Ag_EU_MIMO.csv --dataset_name 09102_Ag_EU_MIMO --n_objects 100000 &> data_harvesting.out &`

`nohup python3.8 download_images.py --saving_path /home/jcejudo/projects/duplicate_detection/data/09102_Ag_EU_MIMO --csv_path /home/jcejudo/projects/duplicate_detection/data/09102_Ag_EU_MIMO.csv &> download_images.out &`


# 90402_M_NL_Rijksmuseum

## Data Harvesting

`nohup python3.8 data_harvesting.py --saving_path /home/jcejudo/projects/duplicate_detection/data/90402_M_NL_Rijksmuseum.csv --dataset_name 90402_M_NL_Rijksmuseum --n_objects 10000000 &> data_harvesting.out &`

`nohup python3.8 download_images.py --saving_path /rnd/duplicate_detection/data/90402_M_NL_Rijksmuseum --csv_path /home/jcejudo/projects/duplicate_detection/data/90402_M_NL_Rijksmuseum.csv &> download_images_rijkmuseum.out &`


to do

https://askubuntu.com/questions/217764/argument-list-too-long-when-copying-files

## Apply duplicate detection to union of datasets

copy images to the same directory

`nohup python3.8  run_duplicate_detection.py --input_dir /home/jcejudo/projects/duplicate_detection/data/90402_M_NL_Rijksmuseum --work_dir /home/jcejudo/projects/duplicate_detection/results/90402_M_NL_Rijksmuseum --num_images 500 --num_components 50 &> duplicate_detection.out &`


include dataset in the final sheet













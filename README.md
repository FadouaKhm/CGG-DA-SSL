# CGG-DA-SSL
Confidence Guided Generative Data Augmentation for semi-supervised Training

## QUICKSTART

0. Run conda env create -f environment.yml for environment initialization.

1. Have your ATR data extracted and saved as .hdf5 files under --data_files_dir (default folder is ./data):
(In my case I have a hdf5 file extracted and saved for each range for both YOLO and GT)

2. Split your hdf5 files into: training, testing and validation and save the list of each categorie's filenames under ./inputs folder:
 - In l_train.txt list your labeled training filenames
 - In val.txtlist your validation filenames ... etc
 - ul_train.txt contains the unlabeled filenames that will be generated afterwards

3. Run the fullysupervised.py model: 
python fullysupervised.py --num_classes 2 --save_name 'FS_output' --dataset 'ATR' --net 'vgg16_bn' --net_from_name True --lr 1e-3 --l_train "./inputs/l_train.txt" --val "./inputs/val.txt" --data_files_dir './data'

(Check arg parser for more options)
The output will be saved under ./saved_models

4. Run 
python generate_ssl_data.py --load_path './saved_models/FS_output/model_best.pth' --val "./inputs/val.txt" --exp_id 'demo' --num_epochs 100 --num_samples 5000

 - This will generate --num_samples unlabeled data, and a labeled reconstructed set
 - Both sets will be saved under ./data
 - the generated data will be used as additional data to train SSL model, make sure to update the txt files under ./inputs before launching the SSL training

5. Run SSL
python mixmatch.py --num_classes 2 --save_name 'MM_output' --dataset 'ATR' --net 'vgg16_bn' --net_from_name True --lr 1e-3 --l_train "./inputs/l_trainSSL.txt" --ul_train "./inputs/ul_train.txt" --val "./inputs/val.txt" --data_files_dir './data'

(Check arg parser for more options)

6. To test:
python eval.py --load_path './saved_models/FS_output/model_best.pth' --test "./inputs/test.txt" 

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

CONFIG="PATH_TO_DESIGNATED_SETTING"

python src/generate_data_ampere_eae.py -c $CONFIG
python src/train_ampere_eae.py -c $CONFIG

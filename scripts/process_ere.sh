export ERE_PATH="../../Dataset/ERE_EN/"
export OUTPUT_PATH="./processed_data/ere_bart"

mkdir $OUTPUT_PATH

python preprocessing/process_ere.py -i $ERE_PATH -o $OUTPUT_PATH -s resource/splits/ERE-EN -b facebook/bart-large -w 1

# add AMR
python preprocessing/add_amr.py -i $OUTPUT_PATH/train.w1.oneie.json -o $OUTPUT_PATH/train.w1.oneie.json
python preprocessing/add_amr.py -i $OUTPUT_PATH/dev.w1.oneie.json -o $OUTPUT_PATH/dev.w1.oneie.json
python preprocessing/add_amr.py -i $OUTPUT_PATH/test.w1.oneie.json -o $OUTPUT_PATH/test.w1.oneie.json

# split low resource
export SPLIT_PATH="./resource/low_resource_split/ere"
python preprocessing/split_dataset.py -i $OUTPUT_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_005 -o $OUTPUT_PATH/train.005.w1.oneie.json
python preprocessing/split_dataset.py -i $OUTPUT_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_010 -o $OUTPUT_PATH/train.010.w1.oneie.json
python preprocessing/split_dataset.py -i $OUTPUT_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_020 -o $OUTPUT_PATH/train.020.w1.oneie.json
python preprocessing/split_dataset.py -i $OUTPUT_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_030 -o $OUTPUT_PATH/train.030.w1.oneie.json    
python preprocessing/split_dataset.py -i $OUTPUT_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_050 -o $OUTPUT_PATH/train.050.w1.oneie.json      
export DYGIEFORMAT_PATH="./processed_data/ace05e_dygieppformat"
export OUTPUT_PATH="./processed_data/ace05e_bart"

mkdir $OUTPUT_PATH

python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/train.json -o $OUTPUT_PATH/train.w1.oneie.json -b facebook/bart-large -w 1
python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/dev.json -o $OUTPUT_PATH/dev.w1.oneie.json -b facebook/bart-large -w 1
python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/test.json -o $OUTPUT_PATH/test.w1.oneie.json -b facebook/bart-large -w 1

# add AMR
python preprocessing/add_amr.py -i $OUTPUT_PATH/train.w1.oneie.json -o $OUTPUT_PATH/train.w1.oneie.json
python preprocessing/add_amr.py -i $OUTPUT_PATH/dev.w1.oneie.json -o $OUTPUT_PATH/dev.w1.oneie.json
python preprocessing/add_amr.py -i $OUTPUT_PATH/test.w1.oneie.json -o $OUTPUT_PATH/test.w1.oneie.json

# split low resource
export SPLIT_PATH="./resource/low_resource_split/ace05e"
python preprocessing/split_dataset.py -i $OUTPUT_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_005 -o $OUTPUT_PATH/train.005.w1.oneie.json
python preprocessing/split_dataset.py -i $OUTPUT_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_010 -o $OUTPUT_PATH/train.010.w1.oneie.json
python preprocessing/split_dataset.py -i $OUTPUT_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_020 -o $OUTPUT_PATH/train.020.w1.oneie.json
python preprocessing/split_dataset.py -i $OUTPUT_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_030 -o $OUTPUT_PATH/train.030.w1.oneie.json    
python preprocessing/split_dataset.py -i $OUTPUT_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_050 -o $OUTPUT_PATH/train.050.w1.oneie.json

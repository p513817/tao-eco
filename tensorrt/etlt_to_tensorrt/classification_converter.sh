#!/bin/bash

function print_div(){
	COL=$(stty size | cut -d" " -f2)
	printf "=%.0s" $(seq 1 $COL)
}

# Help Information
Help()
{
   # Display Help
   echo
   echo "Convert Classification Model (Resnet18) from 'etlt' to 'trt' ('engine')."
   echo
   echo "Syntax: classification_converter.sh [input] [output]"
   echo "	input		path of input etlt model"
   echo "	output		path of output tensorrt engine"
   echo
}

while getopts ":h" option; do
	case $option in
		h)
			Help
			exit;;
	esac
done

#################################################

# File Path
ETLT="$(realpath $1)"
SAVE_TRT="$(realpath -m $2)"

print_div
echo "Input model.etlt : $ETLT"
echo "Output model.trt : $SAVE_TRT"

# Model Opt
KEY=nvidia_tlt
IN_SIZE=3,224,224
IN_ORDER=nchw
OUT_LR=predictions/Softmax

print_div
echo "Model Info:"
echo "Key=$KEY"
echo "Input Size: $IN_SIZE"
echo "Input Order: $IN_ORDER"
echo "Output Layer Name: $OUT_LR"

# Train Opt
MAX_BATCH=4
TYPE=fp32
SPACE=1610612736

print_div
echo "Convert Info:"
echo "Max Batch Size: $MAX_BATCH"
echo "Data Type: $TYPE"
echo "Work Space Size: $SPACE"

print_div
echo "Convert trt from etlt:"
./tlt-converter $ETLT \
-k $KEY -o $OUT_LR \
-d $IN_SIZE -i $IN_ORDER -m $MAX_BATCH -t $TYPE \
-b 1 -w $SPACE \
-e $SAVE_TRT

if [ -f $SAVE_TRT ];then
	echo "Convert Successful "
else
   echo "Convert Failed"
   exit 1
fi

echo -e "Do you want to validate tensorrt engine ? [Y/n]: \c"
read OPT

if [[ ${OPT} = "Y" ]];then
   print_div
   echo "Check TensorRT Engine via trtexec:"
   trtexec --loadEngine=$SAVE_TRT --batch=4;
fi

print_div
echo "All Done. Go 'trt_inference' & Play With Your AI Model."

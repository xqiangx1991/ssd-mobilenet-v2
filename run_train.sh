
export OBJECT_DETECTION=path/to/ssd-mobilenet-v2
export PYTHONPATH=${PYTHONPATH}:${OBJECT_DETECTION}

nohup python3 ${OBJECT_DETECTION}/src/train/train.py \
	      --output_path="${OBJECT_DETECTION}/trained_model" \
        --pipeline_config_path="${OBJECT_DETECTION}/src/config/ssd_mobilenet_v2.yaml" &

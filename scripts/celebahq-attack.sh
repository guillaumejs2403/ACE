MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 500 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1"

Q=31  # 39 for age
T=-1

GPU=0
OUTPUT_PATH='path/to/output'
MODELPATH='path/to/ddpm'
DATAPATH='path/to/dataset'
CLASSIFIERPATH="path/to/classifier"

EXPNAME=experiment_name
python main.py $MODEL_FLAGS $SAMPLE_FLAGS --gpu $GPU \
    --num_samples 30 \
    --model_path $MODELPATH \
    --classifier_path $CLASSIFIERPATH \
    --output_path $OUTPUT_PATH \
    --exp_name $EXPNAME \
    --attack_method PGD \
    --attack_iterations 50 \
    --attack_joint True \
    --dist_l1 0.001 \
    --timestep_respacing 25 \
    --sampling_time_fraction 0.2 \
    --sampling_stochastic True \
    --sampling_inpaint 0.15 \
    --label_query $Q \
    --label_target $T \
    --image_size 256 \
    --data_dir $DATAPATH \
    --dataset CelebAHQ

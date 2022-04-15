export CUDA_VISIBLE_DEVICES=0,1,2,3
nvidia-smi
python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
    --config ./configs/Retrieval_flickr.yaml \
    --output_dir output/Retrieval_flickr_rebuild \
    --checkpoint /SISDC_GPFS/Home_SE/zqcao-suda/yx/image-text/data/ALBEF_data/pre.pth \
    --text_encoder /SISDC_GPFS/Home_SE/zqcao-suda/yx/image-text/data/download/bert-base-uncased
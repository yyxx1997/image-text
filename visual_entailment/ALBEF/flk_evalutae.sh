export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
    --config ./configs/Retrieval_flickr.yaml \
    --output_dir output/Retrieval_flickr_eval \
    --checkpoint /data1/yx/suda/image-text/sotas/ALBEF/output/Retrieval_flickr/checkpoint_best.pth \
    --evaluate
export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.launch --nproc_per_node=2 --use_env re_evaluate.py \
    --config ./configs/Retrieval_coco.yaml \
    --output_dir output/Retrieval_coco_eval \
    --checkpoint /data1/yx/suda/image-text/sotas/ALBEF/output/Retrieval_coco/checkpoint_best.pth \
    --evaluate
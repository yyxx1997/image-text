export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 --use_env VE.py \
    --config ./configs/VE_coco_rebuild.yaml \
    --output_dir output/VE_COCO_rebuild \
    --checkpoint /data1/yx/suda/image-text/image-text/visual_entailment/ALBEF/output/VE_SNLI_rebuild/checkpoint_best.pth
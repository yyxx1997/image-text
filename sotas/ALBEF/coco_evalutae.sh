export CUDA_VISIBLE_DEVICES=7
python re_evaluate.py \
    --config ./configs/Retrieval_coco.yaml \
    --output_dir output/Retrieval_coco_eval \
    --checkpoint /data1/yx/suda/image-text/sotas/ALBEF/output/Retrieval_coco/checkpoint_best.pth \
    --evaluate
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 --use_env re_evaluate.py \
    --config ./configs/Retrieval_eval_coco.yaml \
    --output_dir output/infer \
    --checkpoint /data1/yx/suda/image-text/image-text/sotas/ALBEF/output/Retrieval_coco_rebuild/checkpoint_best.pth \
    --evaluate  \
    --test_file /data1/yx/suda/image-text/data/ALBEF/datas/data/coco_test.json
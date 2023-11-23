export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 --use_env re_evaluate.py \
    --config ./configs/Retrieval_eval_f30k.yaml \
    --output_dir output/infer \
    --checkpoint /data1/yx/suda/image-text/image-text/sotas/ALBEF/output/Retrieval_entail_lr_split_flickr_0.3/checkpoint_best.pth \
    --evaluate  \
    --test_file /data1/yx/suda/image-text/data/re_rebuild_data/temp.json
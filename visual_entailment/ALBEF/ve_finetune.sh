export CUDA_VISIBLE_DEVICES=6,7
python -m torch.distributed.launch --nproc_per_node=2 --use_env VE.py \
    --config ./configs/VE.yaml \
    --output_dir output/VE_SNLI_rebuild \
    --checkpoint /data1/yx/suda/image-text/sotas/ALBEF/datas/pre.pth \
    --te_checkpoint /data1/yx/suda/image-text/textual_entailment/logs/xnli/03_28_07_45/checkpoint-50.pth
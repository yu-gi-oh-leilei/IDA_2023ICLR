






# step 1
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'IDA-R101-448' \
--dataset_dir '/media/data/maleilei/MLICdataset/' \
--backbone resnet101 --dataname coco14 --batch-size 128 --print-freq 400 \
--output './checkpoint/IDA/ResNet_448_MSCOCO14/bce_attention_bs128_work1' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3718 \
--gamma_pos 0 --gamma_neg 0 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim AdamW --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-4 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--nheads 4 --use_intervention --heavy \
--early-stop \
--ema-decay 0.9997 --amp \
--seed 1 \
--lr_scheduler OneCycleLR \
--pattern_parameters single_lr \
--gpus 0,1,2,3










# step 1
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc_supcon.py -a 'Q2L-R101-448' \
--dataset_dir '/media/data/maleilei/MLICdataset/' \
--backbone resnet101 --dataname coco14 --batch-size 64 --print-freq 400 \
--output './checkpoint/Q2L/ResNet_448_MSCOCO14/asl_attention_bs64_work1' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3718 \
--gamma_pos 0 --gamma_neg 4 --dtgfl --loss_clip 0 \
--epochs 80 --lr 5e-5 --optim AdamW --pretrained \
--num_class 80 --img_size 448 --weight-decay 5e-3 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 8192 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop \
--ema-decay 0.9998 --amp \
--seed 1 \
--lr_scheduler OneCycleLR \
--pattern_parameters single_lr \
--gpus 2,3








# torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d \
# main_mlc_supcon.py -a 'Q2L-R101-448' \
# --dataset_dir '/media/data/maleilei/MLICdataset/' \
# --backbone resnet101 --dataname coco14 --batch-size 64 --print-freq 400 \
# --output './checkpoint/Q2L/ResNet_448_MSCOCO14/asl_attention_bs64_work1' \
# --world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3718 \
# --loss_mode 'asl' --partial_loss_mode 'selective' \
# --gamma_pos 0 --gamma_neg 1 --gamma_unann 4 \
# --simulate_partial_type 'rps' --simulate_partial_param 0.5 \
#  --likelihood_topk=5 --prior_threshold 0.5 \
# --dtgfl --loss_clip 0 \
# --epochs 80 --lr 1e-4 --optim AdamW --pretrained \
# --num_class 80 --img_size 448 --weight-decay 2e-2 \
# --cutout --n_holes 1 --cut_fact 0.5 --length 224 \
# --hidden_dim 2048 --dim_feedforward 8192 \
# --enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
# --early-stop \
# --ema-decay 0.9998 --amp \
# --seed 1 \
# --lr_scheduler OneCycleLR \
# --pattern_parameters single_lr \
# --gpus 0,1



torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc_supcon.py -a 'Q2L-R101-448' \
--dataset_dir '/media/data/maleilei/MLICdataset/' \
--backbone resnet101 --dataname coco14_csl --batch-size 128 --print-freq 400 \
--output './checkpoint/Q2L/ResNet_448_MSCOCO14/cls_attention_bs128_work2' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3718 \
--loss_mode 'cls' --partial_loss_mode 'selective' \
--gamma_pos 0 --gamma_neg 1 --gamma_unann 4 \
--simulate_partial_type 'rps' --simulate_partial_param 0.5 \
 --likelihood_topk=5 --prior_threshold 0.5 \
--dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim AdamW --pretrained \
--num_class 80 --img_size 448 --weight-decay 2e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 8192 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop \
--ema-decay 0.9997 --amp \
--seed 1 \
--lr_scheduler OneCycleLR \
--pattern_parameters single_lr \
--prior_path '/media/data2/maleilei/MLIC/CDCR/PartialLabelingCSL/outputs/priors/prior_fpc_1000.csv' \
--gpus 0,1,2,3

# TResNet 21K
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc_supcon.py -a 'Q2L-TResL_22k-448' \
--dataset_dir '/media/data/maleilei/MLICdataset/' \
--backbone tresnetl_v2 --dataname coco14_csl --batch-size 112 --print-freq 400 \
--output './checkpoint/Q2L/TResNetV2_448_COCO/cls_attention_bs112_work2' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3718 \
--loss_mode 'cls' --partial_loss_mode 'selective' \
--gamma_pos 0 --gamma_neg 1 --gamma_unann 4 \
--simulate_partial_type 'rps' --simulate_partial_param 0.5 \
 --likelihood_topk=5 --prior_threshold 0.5 \
--dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 80 --img_size 448 --weight-decay 2e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 --orid_norm \
--hidden_dim 2048 --dim_feedforward 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop \
--ema-decay 0.9997 --amp \
--seed 1 \
--lr_scheduler OneCycleLR \
--pattern_parameters single_lr \
--prior_path '/media/data2/maleilei/MLIC/CDCR/PartialLabelingCSL/outputs/priors/prior_fpc_1000.csv' \
--gpus 0,1,2,3
#!/usr/bin/env python
for((i=1;i<=100;i++));  
do   
python /data/yijia/code2/code/train_expVAE.py --epochs 50 --one_class 5 --resume ./ckpt/ae_mvtec_checkpoint --one_class 5 --layer 'encoder.21' --loss 'mean' --ssim True;  
done 


WANDB_NAME = <Your wandb name>
wandb login <Your wandb api>
python main.py --seed 1234 --wandb_name $WANDB_NAME --dataset cifar10_imb_2 --data_dir ./data --metric multi_class --batch_size 100 --later_batch_size 20 --num_batch 10 --embed_model_config dinov2_vits14.json --classifier_model_config bayesian.json --strategy_config random.json --trainer_config cifar10/passive/dinov2_vits14_linear.json

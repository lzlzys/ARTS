import argparse
import os

import numpy as np
import torch
from torch import optim
from transformers import GPT2Tokenizer

from ICLR_datasets import RecWithContrastiveLearningDataset
from ICLR_trainer import ICLRecTrainer
from prompt_generator import TransformerModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from new_PEPLER import ContinuousPromptLearning
import torch.nn as nn
from ICLR_model import SASRecModel
from ICLR_utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed
from SAS_model import SASRec


def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--data_dir", default="/home/t618141/python_code_lzl/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Toys_and_Games", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--model_idx", default=0, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="1", help="gpu_id")

    # data augmentation args
    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="percentage of negative interactions in a sequence - robustness analysis",
    )
    parser.add_argument(
        "--training_data_ratio",
        default=1.0,
        type=float,
        help="percentage of training samples used for training - robustness analysis",
    )
    parser.add_argument(
        "--augment_type",
        default="random",
        type=str,
        help="default data augmentation types. Chosen from: \
                        mask, crop, reorder, substitute, insert, random, \
                        combinatorial_enumerate (for multi-view).",
    )
    parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
    parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
    parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator")

    ## contrastive learning task args
    parser.add_argument(
        "--temperature", default=1.0, type=float, help="softmax temperature (default:  1.0) - not studied."
    )
    parser.add_argument(
        "--n_views", default=2, type=int, metavar="N", help="Number of augmented data for each sequence - not studied."
    )
    parser.add_argument(
        "--contrast_type",
        default="Hybrid",
        type=str,
        help="Ways to contrastive of. \
                        Support InstanceCL and ShortInterestCL, IntentCL, and Hybrid types.",
    )
    parser.add_argument(
        "--num_intent_clusters",
        default="256",
        type=str,
        help="Number of cluster of intents. Activated only when using \
                        IntentCL or Hybrid types.",
    )
    parser.add_argument(
        "--seq_representation_type",
        default="mean",
        type=str,
        help="operate of item representation overtime. Support types: \
                        mean, concatenate",
    )
    parser.add_argument(
        "--seq_representation_instancecl_type",
        default="concatenate",
        type=str,
        help="operate of item representation overtime. Support types: \
                        mean, concatenate",
    )
    parser.add_argument("--warm_up_epoches", type=float, default=0, help="number of epochs to start IntentCL.")
    parser.add_argument("--de_noise", action="store_true", help="whether to de-false negative pairs during learning.")

    # model args
    parser.add_argument("--model_name", default="ICLRec", type=str)
    parser.add_argument("--hidden_size", type=int, default=768, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=1, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--item_size", default=2942, type=int)
    # train args
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=50, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", default=False, action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--cf_weight", type=float, default=0.1, help="weight of contrastive learning task")
    parser.add_argument("--rec_weight", type=float, default=0.6, help="weight of contrastive learning task")
    parser.add_argument("--intent_cf_weight", type=float, default=0.3, help="weight of contrastive learning task")
    parser.add_argument("--hidden_units", default=768, type=int)
    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    #PEPLER
    parser.add_argument('--words', type=int, default=20,
                        help='number of words to generate for each sample')
    parser.add_argument('--data_path', type=str, default='/data/LiuZunLong/Yelp/reviews.pickle',
                        help='path for loading the pickle data')
    parser.add_argument('--index_dir', type=str, default='/data/LiuZunLong/Yelp/1/',
                        help='load indexes')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--prediction_path', type=str, default='/home/t618141/python_code/new_lzl_model/prediction.txt')
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--prompt_generator_lr', type=int, default=0.0000001)
    parser.add_argument('--real', type=int, default=0)
    parser.add_argument('--all_size', default=(2940), type=int)
    parser.add_argument('--d_k', default=64, type=int)
    parser.add_argument('--n_heads', default=1, type=int)
    parser.add_argument('--d_v', default=64, type=int)
    parser.add_argument('--d_ff', default=2048, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument("--args_str", type=str, default='Toys_and_Games')
    parser.add_argument('--user_num', default=2048, type=int)
    parser.add_argument('--item_num', default=2048, type=int)
    parser.add_argument("--amazon", type=bool, default=True, help="if it is amazon dataset")
    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + ".txt"
    print(args.args_str)
    user_seq, seq, max_item, valid_rating_matrix, test_rating_matrix, test_seq, user_num = get_user_seqs(args)#加载ICLR数据
    args.user_num = user_num
    args.item_num = max_item
    args.all_size = max_item + 2
    args.item_size = max_item + 3
    args.mask_id = max_item + 1
    # save model args
    args_str = f"{args.model_name}-{args.args_str}-{args.model_idx}--{args.lr}-{args.prompt_generator_lr}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")



    with open(args.log_file, "a") as f:
        f.write(str(args) + "\n")

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    print(args.checkpoint_path)

    args.checkpoint_path = 'output/ICLRec-Yelp-{}--{}-{}-new.pt'.format(args.lr, args.args_str, args.prompt_generator_lr)

    # training data for node classification
    cluster_dataset = RecWithContrastiveLearningDataset(
        args, user_seq[: int(len(user_seq) * args.training_data_ratio)], seq[: int(len(seq) * args.training_data_ratio)], data_type="train"
    )
    cluster_sampler = SequentialSampler(cluster_dataset)
    cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size, num_workers=0)

    train_dataset = RecWithContrastiveLearningDataset(
        args, user_seq[: int(len(user_seq) * args.training_data_ratio)], seq[: int(len(seq) * args.training_data_ratio)], data_type="train"
    )
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=0)

    eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=0)

    test_dataset = RecWithContrastiveLearningDataset(args, user_seq, seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=0)

    model = SASRecModel(args=args)

    model.load_state_dict(torch.load('/home/t618141/python_code/ICLR/src/output/{}.pt'.format(args.data_name), map_location='cuda'))

    trainer = ICLRecTrainer(model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args)
    early_epoch = 0

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f"Load model from {args.checkpoint_path} for test!")
        scores, result_info = trainer.test(0, args=args, test_seq=test_seq, full_sort=True)

    else:
        print(f"Train ICLRec")
        early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)
        for epoch in range(args.epochs):

            trainer.train(epoch, args, eval_dataloader)
            # evaluate on NDCG@20
            scores, gpt_socre = trainer.valid(epoch, args, full_sort=True, test_seq=test_seq)
            print('score')
            scores = scores[0]

            number = early_stopping(np.array(scores[-1:]), trainer.model, trainer.gpt2_model, gpt_socre)
            early_epoch = number

            print(number)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        trainer.args.train_matrix = test_rating_matrix
        print("---------------Change to test_rating_matrix!-------------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, args=args, full_sort=True, test_seq=test_seq, early_epoch=early_epoch)




    print(args_str)
    print(result_info)
    with open(args.log_file, "a") as f:
        f.write(args_str + "\n")
        f.write(result_info + "\n")









if __name__ == "__main__":
    main()
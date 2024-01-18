from joyful.fusion_methods import AutoFusion
import argparse
import torch
import os
import joyful
import warnings
import pickle

warnings.filterwarnings("ignore")


log = joyful.utils.get_logger()

def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def func(experiment, trainset, devset, testset, model, opt, sched, args):
    args.hidden_size = experiment.get_parameter("HIDDEN_DIM")
    args.seqcontext_nlayer = experiment.get_parameter("SEQCONTEXT")
    args.gnn_nheads = experiment.get_parameter("GNN_HEAD")
    args.learning_rate = experiment.get_parameter("LR")
    args.wp = experiment.get_parameter("WP")
    args.wf = experiment.get_parameter("WF")
    args.use_highway = experiment.get_parameter("HIGHWAY")
    args.class_weight = experiment.get_parameter("CLASS_WEIGHT")
    args.drop_rate = experiment.get_parameter("DROPOUT")
    args.experiment = experiment

    coach = joyful.Coach(trainset, devset, testset, model, opt, sched, args)
    if not args.from_begin:
        ckpt = torch.load(args.model_ckpt)
        coach.load_ckpt(ckpt)
        print("Training from checkpoint...")

    # Train
    log.info("Start training...")
    (
        best_dev_f1,
        best_epoch,
        best_state,
        train_losses,
        dev_f1s,
        test_f1s,
    ) = coach.train()
    return best_dev_f1


def main(args):
    joyful.utils.set_seed(args.seed)

    if args.emotion:
        args.data = os.path.join(
            args.data_dir_path,
            args.dataset,
            "data_" + args.dataset + "_" + args.emotion + ".pkl",
        )
    else:
        if args.transformers:
            args.data = os.path.join(
                args.data_dir_path,
                args.dataset,
                "transformers",
                "data_" + args.dataset + ".pkl",
            )
            print(os.path.join(args.data_dir_path, args.dataset, "transformers"))
        else:
            args.data = os.path.join(
                args.data_dir_path, args.dataset, "data_" + args.dataset + ".pkl"
            )

    # load data
    log.debug("Loading data from '%s'." % args.data)

    data = joyful.utils.load_pkl(args.data)
    log.info("Loaded data.")

    modelF = AutoFusion(1380)

    trainset = joyful.Dataset(data["train"], modelF, True, args)
    devset = joyful.Dataset(data["dev"], modelF, False, args)
    testset = joyful.Dataset(data["test"], modelF, False, args)

    log.debug("Building model...")

    model = joyful.JOYFUL(args).to(args.device)

    opt1 = joyful.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)

    opt1.set_parameters(list(model.parameters()) + list(modelF.parameters()), args.optimizer1)

    opt2 = joyful.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)

    opt2.set_parameters(list(model.parameters()) + list(modelF.parameters()), args.optimizer2)

    sched1 = opt1.get_scheduler(args.scheduler)

    coach = joyful.Coach(trainset, devset, testset, model, modelF, opt1, sched1, args)

    # Train
    log.info("Start training...")
    coach.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="iemocap_4",
        choices=["iemocap", "iemocap_4", "mosei",'meld'],
        help="Dataset name.",
    )
    ### adding other pre-trained text models
    parser.add_argument("--transformers", action="store_true", default=False)

    """ Dataset specific info (effects)
            -> tag_size in joyful.py
            -> n_speaker in joyful.py
            -> class_weights in classifier.py
            -> label_to_idx in Coach.py """

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )

    # Training parameters
    parser.add_argument(
        "--from_begin", action="store_true", help="Training from begin.", default=False
    )
    parser.add_argument("--model_ckpt", type=str, help="Training from a checkpoint.")

    parser.add_argument("--device", type=str, default="cuda", help="Computing device.")
    parser.add_argument(
        "--epochs", default=1, type=int, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument(
        "--optimizer1",
        type=str,
        default="adam",
        choices=["sgd", "rmsprop", "adam", "adamw"],
        help="Name of optimizer.",
    )
    parser.add_argument(
        "--optimizer2",
        type=str,
        default="sgd",
        choices=["sgd", "rmsprop", "adam", "adamw"],
        help="Name of optimizer.",
    )
    parser.add_argument(
        "--scheduler", type=str, default="reduceLR", help="Name of scheduler."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.00003, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="Weight decay."
    )
    parser.add_argument("--drop_rate", type=float, default=0.3, help="Dropout rate.")

    parser.add_argument("--cl_loss_weight", type=float, default=0.2)

    parser.add_argument(
        "--max_grad_value",
        default=-1,
        type=float,
        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""",
    )

    # Model parameters
    parser.add_argument(
        "--wp",
        type=int,
        default=8,
        help="Past context window size. Set wp to -1 to use all the past context.",
    )

    parser.add_argument(
        "--wf",
        type=int,
        default=8,
        help="Future context window size. Set wp to -1 to use all the future context.",
    )

    parser.add_argument("--n_speakers", type=int, default=2, help="Number of speakers.")

    parser.add_argument(
        "--hidden_size", type=int, default=100, help="Hidden size of two layer GCN."
    )

    parser.add_argument(
        "--rnn",
        type=str,
        default="transformer",
        choices=["lstm", "gru", "transformer"],
        help="Type of RNN cell.",
    )
    parser.add_argument(
        "--class_weight",
        action="store_true",
        default=False,
        help="Use class weights in nll loss.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        choices=["relational", "relative", "multi"],
        help="Type of positional encoding",
    )
    parser.add_argument(
        "--trans_encoding",
        action="store_true",
        default=False,
        help="Use dynamic embedding or not",
    )

    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in seqcontext.py"""
    parser.add_argument(
        "--modalities",
        type=str,
        default="atv",
        required=True,
        choices=["a", "t", "v", "at", "tv", "av", "atv"],
        help="Modalities",
    )

    # emotion
    parser.add_argument(
        "--emotion", type=str, default=None, help="emotion class for mosei"
    )

    # Model Architecture changes
    parser.add_argument("--concat_gin_gout", action="store_true", default=False)
    parser.add_argument("--seqcontext_nlayer", type=int, default=4)
    parser.add_argument("--gnn_nheads", type=int, default=4)
    parser.add_argument("--num_bases", type=int, default=7)
    parser.add_argument("--use_highway", action="store_true", default=False)

    # others
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    parser.add_argument("--use_pe_in_seqcontext", action="store_true", default=False)
    parser.add_argument("--tuning", action="store_true", default=False)
    parser.add_argument("--tag", type=str, default="hyperparameters_opt")

    args = parser.parse_args()

    args.dataset_embedding_dims = {
        "iemocap": {
            "a": 50,
            "t": 256,
            "v": 256,
            "at": 512,
            "tv": 1024,
            "av": 768,
            "atv": 1024,
        },
        "iemocap_4": {
            "a": 50,
            "t": 256,
            "v": 256,
            "at": 512,
            "tv": 1024,
            "av": 768,
            "atv": 1024,
        },
        "mosei": {
            "a": 80,
            "t": 768,
            "v": 35,
            "at": 80 + 768,
            "tv": 768 + 35,
            "av": 80 + 35,
            "atv": 80 + 768 + 35,
        },
        "meld": {
            "a": 100,
            "t": 768,
            "v": 512,
            "at": 512,
            "tv": 768 + 512,
            "av": 612,
            "atv": 768,
        },
    }
    main(args)

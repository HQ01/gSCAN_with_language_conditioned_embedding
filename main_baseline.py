import argparse
import os

from torch.optim.lr_scheduler import LambdaLR

from dataloader import dataloader
from model.config import cfg
from model.model import GSCAN_model
from model.utils import *


'''
# import models
def evaluate(data_iterator, model, max_decoding_steps, pad_idx, sos_idx, eos_idx, max_examples_to_evaluate=None):  # \TODO evaluate function might be broken now. This is Ruis' code.
    target_accuracies = []
    exact_match = 0
    num_examples = 0
    correct_terms = 0
    total_terms = 0
    for input_sequence, output_sequence, target_sequence, _, _, aux_acc_target in predict(
            data_iterator=data_iterator, model=model, max_decoding_steps=max_decoding_steps, pad_idx=pad_idx,
            sos_idx=sos_idx, eos_idx=eos_idx, max_examples_to_evaluate=max_examples_to_evaluate):
        # accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
        # accuracy = sequence_accuracy(output_sequence, target_sequence)
        num_examples += output_sequence.shape[0]
        seq_eq = torch.eq(output_sequence, target_sequence)
        mask = torch.eq(target_sequence, pad_idx) + torch.eq(target_sequence, sos_idx)
               # torch.eq(target_sequence, eos_idx)
        seq_eq.masked_fill_(mask, 0)
        total = (~mask).sum(-1).float()
        accuracy = seq_eq.sum(-1) / total
        total_terms += total.sum().data.item()
        correct_terms += seq_eq.sum().data.item()
        exact_match += accuracy.eq(1.).sum().data.item()
        target_accuracies.append(aux_acc_target)
    return (float(correct_terms) / total_terms) * 100, (exact_match / num_examples) * 100, \
            float(np.mean(np.array(target_accuracies))) * 100

'''

def train(train_data_path: str, val_data_paths: dict, use_cuda: bool):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')

    logger.info("Loading Training set...")
    logger.info(cfg.MODEL_NAME)
    train_iter, train_input_vocab, train_target_vocab = dataloader(train_data_path,
                                                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                                                   use_cuda=use_cuda)  # \TODO add k and statistics and shuffling
    val_iters = {}
    for split_name, path in val_data_paths.items():
        val_iters[split_name], _, _ = dataloader(path, batch_size=cfg.VAL_BATCH_SIZE, use_cuda=use_cuda,
                                input_vocab=train_input_vocab, target_vocab=train_target_vocab)

    pad_idx, sos_idx, eos_idx = train_target_vocab.stoi['<pad>'], train_target_vocab.stoi['<sos>'], \
                                train_target_vocab.stoi['<eos>']

    train_input_vocab_size, train_target_vocab_size = len(train_input_vocab.itos), len(train_target_vocab.itos)

    # \TODO decide whether to add sos token and eos token to each input command/target label
    # Each data item dimension

    '''
    Input (command) [0]: batch_size x max_cmd_len       [1]: batch_size x 0 (len for each cmd)
    Situation: batch_size x grid x grid x feat_size
    Target (action) [0]: batch_size x max_action_len    [1]: batch_size x 0 (len for each action sequence)

    max_cmd_len = 6, max_action_len = 16
    '''
    logger.info("Done Loading Training set.")

    # \TODO add statistics for train/val set, see example below
    '''
    logger.info("  Loaded {} training examples.".format(training_set.num_examples))
    logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))
    '''

    # I think this part is basically saving generated vocabs.
    # if generate_vocabularies:
    #     training_set.save_vocabularies(input_vocab_path, target_vocab_path)
    #     logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    logger.info("Loading Dev. set...")

    val_input_vocab_size, val_target_vocab_size = train_input_vocab_size, train_target_vocab_size

    # Shuffle the test set to make sure that if we only evaluate max_testing_examples we get a random part of the set.

    # val_set.shuffle_data()
    logger.info("Done Loading Dev. set.")

    model = GSCAN_model(pad_idx, eos_idx, train_input_vocab_size, train_target_vocab_size, is_baseline=True)

    model = model.cuda() if use_cuda else model


    log_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg.TRAIN.SOLVER.LR,
                                 betas=(cfg.TRAIN.SOLVER.ADAM_BETA1, cfg.TRAIN.SOLVER.ADAM_BETA2))
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: cfg.TRAIN.SOLVER.LR_DECAY ** (t / cfg.TRAIN.SOLVER.LR_DECAY_STEP))

    # Load model and vocabularies if resuming.
    start_iteration = 1
    best_iteration = 1
    best_accuracy = 0
    best_exact_match = 0
    best_loss = float('inf')

    if cfg.RESUME_FROM_FILE:
        assert os.path.isfile(cfg.RESUME_FROM_FILE), "No checkpoint found at {}".format(cfg.RESUME_FROM_FILE)
        logger.info("Loading checkpoint from file at '{}'".format(cfg.RESUME_FROM_FILE))
        optimizer_state_dict = model.load_model(cfg.RESUME_FROM_FILE)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = model.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(cfg.RESUME_FROM_FILE, start_iteration))

    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < cfg.TRAIN.MAX_EPOCH:  # iterations here actually means "epoch"

        # Shuffle the dataset and loop over it.
        # training_set.shuffle_data() \TODO add reshuffle option in iterator
        num_batch = 0
        for x in train_iter:
            # with torch.no_grad():
            #     model.eval()
            #     logger.info("Evaluating..")
            #     # accuracy, exact_match, target_accuracy = evaluate()
            #
            #     accuracy, exact_match, target_accuracy = evaluate(
            #         val_iter, model=model,
            #         max_decoding_steps=30, pad_idx=pad_idx,
            #         sos_idx=sos_idx,
            #         eos_idx=eos_idx,
            #         max_examples_to_evaluate=None)
            #
            #     logger.info("  Evaluation Accuracy: %5.2f Exact Match: %5.2f "
            #                 " Target Accuracy: %5.2f" % (accuracy, exact_match, target_accuracy))
            is_best = False
            model.train()
            target_scores, target_position_scores = model(x.input, x.situation,
                                                          x.target)  # \TODO: model does not output target position prediction

            loss = model.get_loss(target_scores, x.target[0])

            target_loss = 0
            if cfg.AUXILIARY_TASK:
                target_loss = model.get_auxiliary_loss(target_position_scores,
                                                       x.target)  # \TODO x.target currently does not include ground truth target position
            loss += cfg.TRAIN.WEIGHT_TARGET_LOSS * target_loss

            # Backward pass and update model parameters.
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.update_state(is_best=is_best)

            # Print current metrics.
            if num_batch % cfg.PRINT_EVERY == 0:
                accuracy, exact_match = model.get_metrics(target_scores, x.target[0])
                if cfg.AUXILIARY_TASK:
                    auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores,
                                                                             x.target)  # \TODO add ground truth target position into x.target
                else:
                    auxiliary_accuracy_target = 0.
                learning_rate = scheduler.get_lr()[0]
                logger.info("Iteration %08d, loss %8.4f, accuracy %5.2f, exact match %5.2f, learning_rate %.5f,"
                            " aux. accuracy target pos %5.2f" % (training_iteration, loss, accuracy, exact_match,
                                                                 learning_rate, auxiliary_accuracy_target))

            num_batch += 1

            #test code
            # with torch.no_grad():
            #     model.eval()
            #     logger.info("Evaluating..")
            #     # accuracy, exact_match, target_accuracy = evaluate()
            #     test_exact_match = 0
            #     test_accuracy = 0
            #     print(val_iters)
            #     for split_name, val_iter in val_iters.items():
            #         accuracy, exact_match, target_accuracy = evaluate(
            #             val_iter, model=model,
            #             max_decoding_steps=30, pad_idx=pad_idx,
            #             sos_idx=sos_idx,
            #             eos_idx=eos_idx,
            #             max_examples_to_evaluate=None)
            #         if split_name == 'test':
            #             test_exact_match = exact_match
            #             test_accuracy = accuracy
            #
            #         logger.info(" %s Accuracy: %5.2f Exact Match: %5.2f "
            #                     " Target Accuracy: %5.2f " % (split_name, accuracy, exact_match, target_accuracy))
            #     # try:
            #     #     print(val_iters)
            #     #     for split_name, val_iter in val_iters.items():
            #     #         accuracy, exact_match, target_accuracy = evaluate(
            #     #             val_iter, model=model,
            #     #             max_decoding_steps=30, pad_idx=pad_idx,
            #     #             sos_idx=sos_idx,
            #     #             eos_idx=eos_idx,
            #     #             max_examples_to_evaluate=None)
            #     #         if split_name == 'test':
            #     #             test_exact_match = exact_match
            #     #             test_accuracy = accuracy
            #     #
            #     #         logger.info(" %s Accuracy: %5.2f Exact Match: %5.2f "
            #     #                     " Target Accuracy: %5.2f " % (split_name, accuracy, exact_match, target_accuracy))
            #     # except:
            #     #     print("Exception!")
            #
            #     print("reach here")

        if training_iteration % cfg.EVALUATE_EVERY == 0:  # \TODO add evaluation
            with torch.no_grad():
                model.eval()
                logger.info("Evaluating..")
                # accuracy, exact_match, target_accuracy = evaluate()
                test_exact_match = 0
                test_accuracy = 0
                try:
                    for split_name, val_iter in val_iters.items():
                        accuracy, exact_match, target_accuracy = evaluate(
                            val_iter, model=model,
                            max_decoding_steps=30, pad_idx=pad_idx,
                            sos_idx=sos_idx,
                            eos_idx=eos_idx,
                            max_examples_to_evaluate=None)
                        if split_name == 'dev':
                            test_exact_match = exact_match
                            test_accuracy = accuracy

                        logger.info(" %s Accuracy: %5.2f Exact Match: %5.2f "
                                    " Target Accuracy: %5.2f " % (split_name, accuracy, exact_match, target_accuracy))
                except:
                    print("Exception!")

                if test_exact_match > best_exact_match:
                    is_best = True
                    best_accuracy = test_accuracy
                    best_exact_match = test_exact_match
                    model.update_state(accuracy=test_accuracy, exact_match=test_exact_match, is_best=is_best)
                file_name = cfg.MODEL_NAME + "checkpoint.{}th.tar".format(str(training_iteration))
                if is_best:
                    logger.info("saving best model...")
                    model.save_checkpoint(file_name=file_name, is_best=is_best,
                                          optimizer_state_dict=optimizer.state_dict())

        if training_iteration % cfg.SAVE_EVERY == 0:
            logger.info("forcing to save model every several epochs...")
            file_name =  cfg.MODEL_NAME + " checkpoint_force.{}th.tar".format(str(training_iteration))
            model.save_checkpoint(file_name=file_name, is_best=False, optimizer_state_dict = optimizer.state_dict())

        training_iteration += 1  # warning: iteratin represents epochs here
    logger.info("Finished training.")


def main(flags, use_cuda):
    # print(flags)

    # for arg, val in flags.items():
    #     logger.info("{} : {}".format(arg, val))
    # \TODO enable argparser

    if not os.path.exists(cfg.OUTPUT_DIRECTORY):
        os.mkdir(os.path.join(os.getcwd(), cfg.OUTPUT_DIRECTORY))

    # Some checks on the flags
    if cfg.GENERATE_VOCABULARIES:
        assert cfg.INPUT_VOCAB_PATH and cfg.TARGET_VOCAB_PATH, "Please specify paths to vocabularies to save."

    train_data_path = os.path.join(cfg.DATA_DIRECTORY, "train.json")

    test_splits = [
        'situational_1',
        'situational_2',
        'test',
        'visual',
        'visual_easier',
        'dev'
    ]
    val_data_paths = {split_name: os.path.join(cfg.DATA_DIRECTORY, split_name + '.json') for split_name in test_splits}  # \TODO val dataset not exist

    if cfg.MODE == "train":
        train(train_data_path=train_data_path, val_data_paths=val_data_paths, use_cuda=use_cuda)

    # \TODO enable running on test set. See below.

    # elif flags["mode"] == "test":
    #     assert os.path.exists(os.path.join(flags["data_directory"], flags["input_vocab_path"])) and os.path.exists(
    #         os.path.join(flags["data_directory"], flags["target_vocab_path"])), \
    #         "No vocabs found at {} and {}".format(flags["input_vocab_path"], flags["target_vocab_path"])
    #     splits = flags["splits"].split(",")
    #     for split in splits:
    #         logger.info("Loading {} dataset split...".format(split))
    #         test_set = GroundedScanDataset(data_path, flags["data_directory"], split=split,
    #                                        input_vocabulary_file=flags["input_vocab_path"],
    #                                        target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False,
    #                                        k=flags["k"])
    #         test_set.read_dataset(max_examples=None,
    #                               simple_situation_representation=flags["simple_situation_representation"])
    #         logger.info("Done Loading {} dataset split.".format(flags["split"]))
    #         logger.info("  Loaded {} examples.".format(test_set.num_examples))
    #         logger.info("  Input vocabulary size: {}".format(test_set.input_vocabulary_size))
    #         logger.info("  Most common input words: {}".format(test_set.input_vocabulary.most_common(5)))
    #         logger.info("  Output vocabulary size: {}".format(test_set.target_vocabulary_size))
    #         logger.info("  Most common target words: {}".format(test_set.target_vocabulary.most_common(5)))

    #         model = Model(input_vocabulary_size=test_set.input_vocabulary_size,
    #                       target_vocabulary_size=test_set.target_vocabulary_size,
    #                       num_cnn_channels=test_set.image_channels,
    #                       input_padding_idx=test_set.input_vocabulary.pad_idx,
    #                       target_pad_idx=test_set.target_vocabulary.pad_idx,
    #                       target_eos_idx=test_set.target_vocabulary.eos_idx,
    #                       **flags)
    #         model = model.cuda() if use_cuda else model

    #         # Load model and vocabularies if resuming.
    #         assert os.path.isfile(flags["resume_from_file"]), "No checkpoint found at {}".format(flags["resume_from_file"])
    #         logger.info("Loading checkpoint from file at '{}'".format(flags["resume_from_file"]))
    #         model.load_model(flags["resume_from_file"])
    #         start_iteration = model.trained_iterations
    #         logger.info("Loaded checkpoint '{}' (iter {})".format(flags["resume_from_file"], start_iteration))
    #         output_file_name = "_".join([split, flags["output_file_name"]])
    #         output_file_path = os.path.join(flags["output_directory"], output_file_name)
    #         output_file = predict_and_save(dataset=test_set, model=model, output_file_path=output_file_path, **flags)
    #         logger.info("Saved predictions to {}".format(output_file))

    elif cfg.MODE == "predict":
        raise NotImplementedError()


    else:
        raise ValueError("Wrong value for parameters --mode ({}).".format(cfg.MODE))


if __name__ == "__main__":
    FORMAT = "%(asctime)-15s %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                        datefmt="%Y-%m-%d %H:%M")
    logger = logging.getLogger(__name__)
    use_cuda = True if torch.cuda.is_available() else False
    logger.info("Initialize logger")

    if use_cuda:
        logger.info("Using CUDA.")
        logger.info("Cuda version: {}".format(torch.version.cuda))

    parser = argparse.ArgumentParser(description="LGCN models for GSCAN")
    # \TODO merge args into config. See Ronghang's code.
    args = parser.parse_args()

    main(args, use_cuda)

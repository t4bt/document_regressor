import argparse
import os
import multiprocessing
import numpy as np
import chainer
from chainer.backends import cuda
from chainer import training
from chainer import datasets, iterators, optimizers, serializers
import chainer.links as L
from chainer.training import extensions
from sklearn.model_selection import train_test_split
import model

def parse_cmd_args():
    parser = argparse.ArgumentParser(description='Chainer Training:')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of data in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    parser.add_argument('--w_decay', '-w', type=float, default=1e-4,
                        help='Weight Decay')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Number of GPUs')
    parser.add_argument('--out', '-o', default='result/',
                        help='Directory to output the result')
    parser.add_argument('--no_snapshot', action='store_true',
                        help='Suppress storing snapshots.')
    args = parser.parse_args()

    print('# GPUs: {}'.format(args.n_gpus))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    return args

def my_concat_examples(batch, device=None):
    # batch = [(array([0,1,2,3]),      0),
    #          (array([6,3,7,3,2,4]),  1),
    #          (array([6,9,8,2,5])),   2)]
    if device is None:
        x = np.array([i[0] for i in batch])
        t = np.array([i[1] for i in batch])
        return x, t
    elif device < 0:
        x = np.array([i[0] for i in batch])
        t = np.array([i[1] for i in batch])
        return cuda.to_cpu(x), cuda.to_cpu(t)
    else:
        xp = cuda.cupy
        x = [cuda.to_gpu(xp.array(i[0], dtype=xp.int32), device) for i in batch]
        t = xp.array([i[1] for i in batch], dtype=xp.float32)
        return x, cuda.to_gpu(t, device)

def prepare_extensions(trainer, evaluator, args):
    trainer.extend(evaluator)
    trainer.extend(extensions.ExponentialShift('lr', 0.7), trigger=(10, 'epoch'))
    trainer.extend(extensions.dump_graph('main/loss'))

    if not args.no_snapshot:
        trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
            'main/r2', 'validation/main/r2', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
            'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/r2', 'validation/main/r2'],
            'epoch', file_name='r2.png'))
    trainer.extend(extensions.ProgressBar(update_interval=100))

def train_using_gpu(args, model, x, t, valid_rate=0.1, lr=1e-3, weight_decay=1e-3):
    if args.n_gpus == 1:
        print('Start a training script using single GPU.')
    else:
        multiprocessing.set_start_method('forkserver')
        print('Start a training script using multiple GPUs.')

    # Set up a dataset and prepare train/valid data iterator.
    threshold = int(len(t)*(1-valid_rate))
    train = datasets.tuple_dataset.TupleDataset(x[0:threshold], t[0:threshold])
    valid = datasets.tuple_dataset.TupleDataset(x[threshold:], t[threshold:])

    if args.n_gpus == 1:
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    else:
        train_iter = [chainer.iterators.SerialIterator(sub_train, args.batchsize) \
                        for sub_train \
                        in chainer.datasets.split_dataset_n_random(train, args.n_gpus)]
    valid_iter = chainer.iterators.SerialIterator(
                                    valid, args.batchsize, repeat=False, shuffle=False)

    # Make a specified GPU current
    master_gpu_id = 0
    if args.n_gpus == 1:
        chainer.cuda.get_device_from_id(master_gpu_id).use()
        model.to_gpu()  # Copy the model to the GPU
    else:
        chainer.cuda.get_device_from_id(master_gpu_id).use()

    # Make optimizer.
    optimizer = chainer.optimizers.MomentumSGD(lr=lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

    # Set up a trainer
    if args.n_gpus == 1:
        updater = training.StandardUpdater(train_iter, optimizer,
                                            converter=my_concat_examples, device=0)
    else:
        devices_list = {'main': master_gpu_id}
        devices_list.update({'gpu{}'.format(i): i for i in range(1, args.n_gpus)})
        print(devices_list)
        updater = training.updaters.MultiprocessParallelUpdater(train_iter, optimizer,
                                                                converter=my_concat_examples,
                                                                devices=devices_list)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    evaluator = extensions.Evaluator(valid_iter, model,
                                        converter=my_concat_examples, device=master_gpu_id)

    # Set some extension modules to a trainer.
    prepare_extensions(trainer, evaluator, args)

    # Run the training
    trainer.run()

    # Show real throughput.
    datasize = len(train) * args.epoch
    throughput = datasize / trainer.elapsed_time
    print('Throughput: {} [docs/sec.] ({} / {})'.format(
        throughput, datasize, trainer.elapsed_time))

    # Save trained model.
    model_filepath = os.path.join(args.out, 'trained.model')
    chainer.serializers.save_npz(model_filepath, model)

def train_using_cpu(args, model, x, t, valid_rate=0.1, lr=1e-3, weight_decay=1e-4):
    print('Start a training script using single CPU.')

    # Set up a dataset and prepare train/valid data iterator.
    threshold = int(len(t)*(1-valid_rate))
    train = datasets.tuple_dataset.TupleDataset(x[0:threshold], t[0:threshold])
    valid = datasets.tuple_dataset.TupleDataset(x[threshold:], t[threshold:])

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize,
                                                    repeat=False, shuffle=False)

    # Make optimizer.
    optimizer = chainer.optimizers.MomentumSGD(lr=lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, converter=my_concat_examples)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    evaluator = extensions.Evaluator(valid_iter, model, converter=my_concat_examples)

    # Set some extension modules to a trainer.
    prepare_extensions(trainer, evaluator, args)

    # Run the training
    trainer.run()

    # Show real throughput.
    datasize = len(train) * args.epoch
    throughput = datasize / trainer.elapsed_time
    print('Throughput: {} [docs/sec.] ({} / {})'.format(
        throughput, datasize, trainer.elapsed_time))

    # Save trained model.
    model_filepath = os.path.join(args.out, 'trained.model')
    chainer.serializers.save_npz(model_filepath, model)

if __name__ == '__main__':
    args = parse_cmd_args()

    # docs = [array([0,1,2,3]),
    #         array([6,3,7,3,2,4]),
    #         array([6,9,8,2,5])]
    # target = array([0.3,
    #                 0.5,
    #                 0.4])
    # n_words = 10
    # embed_size = 128
    # initialW = None (or Pre-trained WordVector)
    # If you use Pre-trained WordVector (gensim, fasttext)
    # model = gensim.models.KeyedVectors.load_word2vec_format(vector_file)
    # initialW = model.wv.vectors

    X_train, X_test, y_train, y_test = train_test_split(docs, target, test_size=0.1, random_state=42)
    print("train_size:{}".format(len(X_train)))
    print("test_size:{}".format(len(X_test)))

    settings = {
        "n_words":n_words,
        "embed_size":embed_size,
        "hidden_size":128,
        "n_layers":1,
        "initialW":initialW
    }
    Model = model.MyRegressor(model.BiLSTMwithAttentionRegressor(**settings))
    if args.n_gpus > 0:
        train_using_gpu(args, Model, X_train, y_train,
                        lr=args.learnrate, weight_decay=args.w_decay)
    else:
        train_using_cpu(args, Model, X_train, y_train,
                        lr=args.learnrate, weight_decay=args.w_decay)

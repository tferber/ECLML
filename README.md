# ECLML

Graph Net clustering for the Belle II electromagnetic calorimeter.

### Usage

`usage: gravnet_1.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                    [--trainfrac TRAINFRAC] [--ncpu NCPU] [--seed SEED]
                    [--modeldir MODELDIR] [--inferonly] [--nsave NSAVE]
                    [--ninference NINFERENCE] [--inferencedir INFERENCEDIR]
                    [--overtrain] [--nodecay] [--refresh] [--debug-train]
                    [--debug-test] [--no-test] [--print-model] [--use-cpu]

PyTorch ECL ML clustering

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        input batch size for training (default: 128)
  --epochs EPOCHS       number of epochs (default: 1)
  --trainfrac TRAINFRAC
                        fraction of events used for training (default: 0.9)
  --ncpu NCPU           how many CPUs are used in loaders (default: 1)
  --seed SEED           set random seed for all random generators
  --modeldir MODELDIR   directory with pretrained model
  --inferonly           only run inference, no train or test.
  --nsave NSAVE         save model and status every nsave epochs
  --ninference NINFERENCE
                        save inference ntuples for one batch every ninference
                        epochs
  --inferencedir INFERENCEDIR
                        directory to store inference ntuples for one batch
  --overtrain           only use one event to force overtraining
  --nodecay             do not decay learning rate (LR)
  --refresh             do not load prepocessed datasets.
  --debug-train         print loss for every training batch
  --debug-test          print loss for every test batch
  --no-test             skip testing
  --print-model         print model
  --use-cpu             do not use GPU even if it is available`


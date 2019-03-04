## Decision Trees README

Group Project for my MSc Machine Learning Module. The aim was to create decision trees to predict which room a person is in depending on the strengths of wifi signals. We created, evaluated and then optimised the trees.


### Special Mentions:
1. **There is a special comment where to replace the input data file: search in main.py 'DEFAULT INPUT FILE EDIT HERE TO LOAD CUSTOM DEFAULT DATA'**
1. **There are sample diagrams in the out/ folder**
3. A diagram is produced only when running in single mode, which is the default modes
4. There are sample diagram in the out/ folder of how the trees should look for the clean an noisy dataset_copy


### Help:
-python3 main.py <-h or --help> for help
-python3 main.py <-i or --input=> <filepath> to set input file
-python3 main.py <-c or --clean> to run on the clean dataset, this is the default running mode
-python3 main.py <-n or --noisy> to run on the noisy dataset
-python3 main.py <-r or --runmode=> <single or kfold or kfoldall> to run in single or kfold mode. Default mode is single
-Running kfoldall runs all four possible combination of clean or noisy and pruning or no pruning
-python3 main.py <-p or --prune> to enable pruning. Default is off, single mode option only. Saves to diagrams
-A diagram is produced only when running in single mode at the following location ../out/diagram.pdf
-python3 main.py <--drawoff> to disable diagram rendering
-\nSample commanand:\npython main.py --n --runmode=single --shuffleoff

### How to run:
1. Install python requirements `pip3 install -r requirements`
2. Go to source directory `cd src`
3. Run the main program `python3 main.py`

### Useful commands:
1. `python3 main.py --clean --prune --runmode=single` To run on the clean dataset with pruning enabled
2. `python3 main.py -i <filepath>` To load and run on custom dataset_copy
3. `python3 main.py --runmode=kfoldall` To run on all possible combinations

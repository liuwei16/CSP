################################
This script is created to produce miss rate numbers by making minor changes to the COCO python evaluation script [1]. It is a python re-implementation of the Caltech evaluation code, which is written in matlab. This python script produces exactly the same numbers as the matlab code.
[1] https://github.com/pdollar/coco/tree/master/PythonAPI
[2] http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/code/code3.2.1.zip
#################################
Usage
1. Prepare detection results in COCO format, and write them in a single .json file.
2. Run eval_demo.py.
3. Detailed evaluations will be written to results.txt.
#################################


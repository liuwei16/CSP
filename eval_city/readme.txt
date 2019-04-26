Submitting results
Please write the detection results for all test images in a single .json file (see dt_mat2json.m), and then send to shanshan.zhang@njust.edu.cn.
You'll receive the evaulation results via e-mail, and then you decide whether to publish it or not.
If you would like to publish your results on the dashboard, please also specify a name of your method.

Metrics
We use the same protocol as in [1] for evaluation. As a numerical measure of the performance, log-average miss rate (MR) is computed by averaging over the precision range of [10e-2; 10e0] FPPI (false positives per image). 
For detailed evaluation, we consider the following 4 subsets:
1. 'Reasonable': height [50, inf]; visibility [0.65, inf]
2. 'Reasonable_small': height [50, 75]; visibility [0.65, inf]
3. 'Reasonable_occ=heavy': height [50, inf]; visibility [0.2, 0.65]
4. 'All': height [20, inf]; visibility [0.2, inf]


Reference
[1] P. Dollar, C. Wojek, B. Schiele and P. Perona. Pedestrian Detection: An Evaluation of the State of the Art. TPAMI, 2012. 

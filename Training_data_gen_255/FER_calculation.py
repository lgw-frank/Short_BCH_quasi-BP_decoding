# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 07:39:41 2023

@author: lgw
"""
#CCSDS(128,64)  2.0-3.6-0.2dB, NMS-2
ldpc_testing_fer = [(2.0,0.507948),(2,2,0.419679),(2.4,0.336529),(2.6,0.258148),(2.8,0.191503),(3.0,0.133917),(3.2,0.090387),(3.4,0.057173),(3.6,0.034405)]
#CCSDS(128,64) 2.0-3.0-0.2dB, non-DIA,order-3,fixed
osd_fer1 = [(2.0,0.0394),(2.2,0.0253),(2.4,0.0175),(2.6,0.0149),(2.8,0.0105),(3.0,0.00575)]
#CCSDS(128,64) 2.0-3.0-0.2dB, DIA,order-3,fixed
osd_fer2 = [(2.0,0.0233),(2.2,0.015875),(2.4,0.009),(2.6,0.0069),(2.8,0.0033),(3.0,0.00215)]
#CCSDS(128,64) 2.6-3.2-0.2dB, non-DIA,order-3,dynamic
#candidate list size: 23708,23491,23277,23041,22721,22439
osd_fer3 = [(2.0,0.0447),(2.2,0.0342),(2.4,0.0239),(2.6,0.0172) ,(2.8,0.0123),(3.0,0.0096),(3.2,0.0089),(3.4,0.0058),(3.6,0.0051)]
#CCSDS(128,64) 2.0-3.0-0.2dB DIA,order-3,dynamic
#candidate list size: 13442,13376,13455
osd_fer4 = [(2.0,0.0324 ),(2.2,0.02),(2.4,0.0122),(2.6,0.0084),(2.8,0.0056),(3.0,0.0029)]

osd_fer_order_3_list = [osd_fer1,osd_fer2,osd_fer3,osd_fer4]
order_3_fer_list = []
for k in range(4):
    order_3_fer = [round(ldpc_testing_fer[i][1]*osd_fer_order_3_list[k][i][1],4) for i in range(len(ldpc_testing_fer[:6]))]
    order_3_fer_list.append(order_3_fer)
    print(order_3_fer)
    
#CCSDS(128,64) 2.6-3.6-0.2dB non-DIA,order-2,fixed
osd_fer5 = [(2.6,0.0417),(2.8,0.0283),(3.0,0.0218),(3.2,0.022),(3.4,0.0163),(3.6,0.0121)]
#CCSDS(128,64) 2.6-3.6-0.2dB DIA,order-2,fixed
osd_fer6 = [(2.6,0.0221),(2.8,0.0135),(3.0,0.0083),(3.2,0.0055),(3.4,0.0049),(3.6,0.0029)]
#CCSDS(128,64) 2.6-3.2-0.2dB, non-DIA,order-3,dynamic
osd_fer7 = [(2.6,0.0423),(2.8,0.0289),(3.0,0.022),(3.2,0.0231),(3.4,0.0169),(3.6,0.0129)]
#CCSDS(128,64) 2.6-3.6-0.2dB DIA,order-2,dynamic
osd_fer8 = [(2.6,0.0225),(2.8,0.0136),(3.0,0.0086),(3.2,0.0055),(3.4,0.0049),(3.6,0.0029)]
osd_fer_order_2_list = [osd_fer5,osd_fer6,osd_fer7,osd_fer8]

order_2_fer_list = []
for k in range(4):
    order_2_fer = [round(ldpc_testing_fer[i+3][1]*osd_fer_order_2_list[k][i][1],4) for i in range(len(ldpc_testing_fer[3:]))]
    order_2_fer_list.append(order_2_fer)
    print(order_2_fer)

###########################order-3##########################################
#CCSDS(128,64) 2.0-3.0-0.2dB, non-DIA,order-3,fixed
# For 2.0dB (order_sum:3) summary:
# ----> S:4899 F:201
# FER:0.0394 Average candidate size:43745.0
# For 2.2dB (order_sum:3) summary:
# ----> S:7700 F:200
# FER:0.0253 Average candidate size:43745.0
# For 2.4dB (order_sum:3) summary:
# --> S/F: 7860 / 140 Average size: 43745.0
# For 2.6dB (order_sum:3) summary:
# --> S/F: 11821 / 179 Average size: 43745.0
# For 2.8dB (order_sum:3) summary:
# --> S/F: 8905 / 95 Average size: 43745.0
# For 3.0dB (order_sum:3) summary:
# --> S/F: 7954 / 46  Average size: 43745.0

# #CCSDS(128,64) 2.0-3.0-0.2dB, DIA,order-3,fixed
# For 2.0dB (order_sum:3) summary:
# ----> S:8400 F:200
# FER:0.0233 Average candidate size:43745.0
# For 2.2dB (order_sum:3) summary:
# --> S/F: 7873 / 127  Average size: 43745.0
# For 2.4dB (order_sum:3) summary:
# --> S/F: 16846 / 154  Average size: 43745.0
# For 2.6dB (order_sum:3) summary:
# --> S/F: 8938 / 62 Average size: 43745.0
# For 2.8dB (order_sum:3) summary:
# --> S/F: 9967 / 33  Average size: 43745.0
# For 3.0dB (order_sum:3) summary:
# --> S/F: 11974 / 26  Average size: 43745.0

# #CCSDS(128,64) 2.6-3.2-0.2dB, non-DIA,order-3,dynamic
# For 2.0dB (order_sum:3) benchmark-dynamic:
# ----> S:4299 F:201
# FER:0.0447 Average candidate size: 24263.9527
# For 2.2dB (order_sum:3) summary:
# ----> S:5698 F:202
# FER:0.0342 Average candidate size:24144.0536
# For 2.4dB (order_sum:3) summary:
# ----> S:8199 F:201
# FER:0.0239 Average candidate size:23977.3589
# For 2.6dB (order_sum:3) summary:
# ----> S:25567 F:447
# FER:0.0172 Average candidate size:23708.6651
# For 2.8dB (order_sum:3) summary:
# ----> S:22525 F:281
# FER:0.0123 Average candidate size:23491.5317
# For 3.0dB (order_sum:3) summary:
# ----> S:15916 F:154
# FER:0.0096 Average candidate size:23277.1724
# For 3.2dB (order_sum:3) summary:
# ----> S:13438 F:120
# FER:0.0089 Average candidate size:23041.983
# For 3.4dB (order_sum:3) summary:
# ----> S:8526 F:50
# FER:0.0058 Average candidate size:22721.7825
# For 3.6dB (order_sum:3) summary:
# ----> S:6846 F:35
# FER:0.0051 Average candidate size:22439.3058

# #CCSDS(128,64) 2.6-3.2-0.2dB, DIA,order-3,dynamic
# For 2.0dB (order_sum:3) model_cnn-dynamic:
# ----> S:5999 F:201
# FER:0.0324 Average candidate size: 13290.3511 
# For 2.2dB (order_sum:3) summary:
# --> S/F: 8820 / 180  Average size: 13365.1762
# For 2.4dB (order_sum:3) model_cnn-dynamic:
# ----> S:16299 F:201
# FER:0.0122 Average candidate size: 13443.4167 
# For 2.6dB (order_sum:3) summary:
# ----> S:25796 F:218
# FER:0.0084 Average candidate size:13442.3844
# For 2.8dB (order_sum:3) dynamicmodel_cnn:
# ----> S:22678 F:128
# FER:0.0056 Average candidate size: 13376.605 
# For 3.0dB (order_sum:3) summary:
# ----> S:16024 F:46
# FER:0.0029 Average candidate size:13455.4973

# ###########################order-2##########################################
# #CCSDS(128,64) 2.6-3.6-0.2dB non-DIA,order-2,fixed
# For 2.6dB (order_sum:2) summary:
# ----> S:4600 F:200
# FER:0.0417 Average candidate size:2081.0
# For 2.8dB (order_sum:2) summary:
# ----> S:6899 F:201
# FER:0.0283 Average candidate size:2081.0
# For 3.0dB (order_sum:2) summary:
# ----> S:8999 F:201
# FER:0.0218 Average candidate size:2081.0
# For 3.2dB (order_sum:2) summary:
# ----> S:8900 F:200
# FER:0.022 Average candidate size:2081.0
# For 3.4dB (order_sum:2) summary:
# ----> S:8436 F:140
# FER:0.0163 Average candidate size:2081.0
# For 3.6dB (order_sum:2) summary:
# ----> S:6798 F:83
# FER:0.0121 Average candidate size:2081.0

# #CCSDS(128,64) 2.6-3.6-0.2dB DIA,order-2,fixed
# For 2.6dB (order_sum:2) summary:
# ----> S:8899 F:201
# FER:0.0221 Average candidate size:2081.0
# For 2.8dB (order_sum:2) summary:
# ----> S:14600 F:200
# FER:0.0135 Average candidate size:2081.0
# For 3.0dB (order_sum:2) summary:
# ----> S:15936 F:134
# FER:0.0083 Average candidate size:2081.0
# For 3.2dB (order_sum:2) summary:
# ----> S:13484 F:74
# FER:0.0055 Average candidate size:2081.0
# For 3.4dB (order_sum:2) summary:
# ----> S:8534 F:42
# FER:0.0049 Average candidate size:2081.0
# For 3.6dB (order_sum:2) summary:
# ----> S:6861 F:20
# FER:0.0029 Average candidate size:2081.0

# #CCSDS(128,64) 2.6-3.6-0.2dB non-DIA,order-2,dynamic
# osd_fer3 = [0.0385,0.0263,0.0224,0.0227,0.0168,0.0121]

# For 2.6dB (order_sum:2) summary:
# ----> S:4597 F:203
# FER:0.0423 Average candidate size:1910.1402
# For 2.8dB (order_sum:2) summary:
# ----> S:6798 F:202
# FER:0.0289 Average candidate size:1896.1763
# For 3.0dB (order_sum:2) summary:
# ----> S:8998 F:202
# FER:0.022 Average candidate size:1883.9201
# For 3.2dB (order_sum:2) summary:
# ----> S:8499 F:201
# FER:0.0231 Average candidate size:1879.3506
# For 3.4dB (order_sum:2) summary:
# ----> S:8431 F:145
# FER:0.0169 Average candidate size:1854.3572
# For 3.6dB (order_sum:2) summary:
# ----> S:6792 F:89
# FER:0.0129 Average candidate size:1843.9965

# #CCSDS(128,64) 2.6-3.6-0.2dB DIA,order-2,dynamic
# For 2.6dB (order_sum:2) summary:
# ----> S:8700 F:200
# FER:0.0225 Average candidate size:1822.9436
# For 2.8dB (order_sum:2) summary:
# ----> S:14500 F:200
# FER:0.0136 Average candidate size:1813.4623
# For 3.0dB (order_sum:2) summary:
# ----> S:15932 F:138
# FER:0.0086 Average candidate size:1813.0834
# For 3.2dB (order_sum:2) summary:
# ----> S:13483 F:75
# FER:0.0055 Average candidate size:1804.9164
# For 3.4dB (order_sum:2) summary:
# ----> S:8534 F:42
# FER:0.0049 Average candidate size:1786.0169
# For 3.6dB (order_sum:2) summary:
# ----> S:6861 F:20
# FER:0.0029 Average candidate size:1783.8872






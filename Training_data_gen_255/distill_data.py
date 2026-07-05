# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 19:18:02 2022

@author: zidonghua_30
"""

import re
import os


# root_dir = r'D:\tmp'
# file_result = []
# for dir in os.listdir(root_dir):
# 	file_result.append(dir)
# 	
# print(file_result)

# import os

# root_dir = r'D:\tmp'
# file_result = []
# for root, dirs, files in os.walk(root_dir ):
#         # 遍历输出所有文件路径
#         for name in files:
#             file_result.append(os.path.join(root_dir , name))
#         for name in dirs:
#             file_result.append(os.path.join(root_dir , name))
            
# print(file_result)  

# import os

# root_dir = r'D:\tmp'
# file_result = []
# for dir in os.listdir(root_dir):
#     child = os.path.join(root_dir, dir)
#     if dir.endswith(".txt"):
#         file_result.append(dir)
#     if os.path.isdir(child):
#         for f1 in os.listdir(child):
#             if f1.endswith(".txt"):
#                 file_result.append(f1)
                
# print(file_result)


output_file = 'testing-1008-whole-fer'
num_regex = re.compile(r'[.0-9]*')
alphabet_regex = re.compile(r'[a-zA-z]')
character_regex = re.compile(r'[\u4E00-\u9FA5]')
#root_dir = r'./SPA_FER_OVER/'
root_dir = r'./SPA_FER_UNDER/'
filename = "log"
file_result = []
for root, dirs, files in os.walk(root_dir):
    for name in files:
        if filename in name:
            file_result.append(name)
#print(file_result)
output_file = root_dir + output_file
with open(output_file, 'w', encoding='utf-8') as file_writer:
    for file_feeding in file_result:
        file_feeding = root_dir + file_feeding
        with open(file_feeding, 'r', encoding='utf-8') as file_reader:
            print("file name:",file_feeding)
            file_writer.write("file name:"+file_feeding+'\n')
            str_data = file_reader.readlines()
            for line in str_data:
                #line_seg = line.split()
                index = re.findall(r"pre:", line)
                if index:
                    index2 = num_regex.findall(line)
                    new_list = [i for i in index2 if i !='']
                    new_string = '('+new_list[2]+','+new_list[3]+')'       
                    print(new_string)
                    file_writer.write(new_string+'\n')
    



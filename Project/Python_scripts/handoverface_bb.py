#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 20:00:26 2022

@author: violo
"""
# example of extracting bounding boxes from an annotation file
from xml.etree import ElementTree
import os
import xml.dom.minidom


def extract_handoverface(pos_xml , bounded , name):
    #read file xml
    with open(pos_xml + '/' + str(name) +'.xml','r') as t:
        data = t.read()
    #analyze all polygon into xml file    
    poly_split = data.split("<polygon>")
    poly_split.pop(0)
    for elem in poly_split :
        x_elem = elem.split("</x>")
        x_max = 0
        x_min = 3000
        #analyze all x element into polygon for find maximum and minimum
        for f_elem in x_elem :
            value = f_elem.split("<x>")
            value.pop(0)
            if len(value) != 0:
                n_value = int(value[0])
                if x_max < n_value:
                    x_max=n_value
                if x_min > n_value:
                    x_min=n_value               
        y_elem = elem.split("</y>")
        y_max = 0
        y_min = 3000
        #analyze all y into polygon for find the minimun and maximum
        for f_elem in y_elem :
            value = f_elem.split("<y>")
            value.pop(0)
            if len(value) != 0:
                n_value = int(value[0])
                if y_max < n_value:
                    y_max=n_value
                if y_min > n_value:
                    y_min=n_value
        #write bounded box into the format x-y-width-height            
        file_txt = open(bounded + "/" + str(name) +".txt", "a")            
        width = x_max-x_min
        height = y_max-y_min
        file_txt.write(str(x_min) + " " + str(y_min) + " " + str(width) + " " + str(height) + "\n")
        file_txt.close()            
#position file xml
pos_xml = r"hand/annotations" 
#position where save the bounded box
bounded = r"hand/bounded"
for i in range(1,302):
    if not (i==133 or i==166):
        extract_handoverface(pos_xml , bounded , i)               
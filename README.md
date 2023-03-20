# ENPM661_Project3_phase1
Github repo for the codes, documentation, and results for ENPM661: Planning for autonomous robots Project 3 Phase 1

## Github link:
https://github.com/muditsingal/ENPM661_Project3_phase1

### General note:
I have used a hash map to store the node and their parameters as per the pixels in the canvas. This reduces the time required to check if an element is in open list from O(n) to O(1), which improves performance by upto 20 times. The data structure used in an 3D array where the first 2 dimensions represent the y and x coordinates of the image pixel locations, and the 3rd dimension represents the parameters of each node element, such as ctc, is_opened, is_closed and the node object itself.

### Libraries:
OpenCV, Numpy, time


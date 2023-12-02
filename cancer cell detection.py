#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


import cv2


# In[10]:


path = r'C:\Users\HP\Desktop\kkk.jpg'


# In[11]:


os.path.exists(path)


# In[12]:



img = cv2.imread(path)
#img = cv2.resize(img, (200,200))
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0)


# In[13]:


# Change the values of Threshold for further fine tuning
ret, thresh = cv2.threshold(imgBlur, 140, 190, 0)

# Create image copy to draw outline for cancer cells
img_res = img.copy()

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_cancer = cv2.drawContours(img, contours, -1, (125,125,0), 2)


# In[ ]:



cv2.imshow('output', img)
cv2.imshow('input', img_res)
cv2.imshow('cancer cell detection',img_cancer+img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





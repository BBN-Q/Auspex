# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

"""
Created on Wed Mar 21 11:35:40 2012
Some convenience functions for uploading figures to the wiki.  
These are mainly built around dumping figures to a particular day-to-day folder for
lab-book entries. 
@author: Colm Ryan
"""

import os, sys
from xmlrpclib import ServerProxy, Binary
import time
import urllib
import argparse


def make_daily_folder(ploneServer, baseFolder, folderDate):
    '''
    If necessary create the folder for the desired date.
    Getting is info is a bit obtuse:
        get_object returns a dictionary because it can return content for multiple items
        Each object of the dictionary is a list of [schema, type, extensions]
        Since we are looking at folders, extensions is a dictionary with a 'contents' keys
        The 'contents' value returns another dictionary with a a set of info.
    '''
    
    #Check whether we have the year folder in the base folder
    tmpContent = ploneServer.get_object([baseFolder]).values()[0]
    curYears = [x['Title'] for x in tmpContent[2]['contents'].values()]
    yearStr = folderDate.strftime('%Y')
    if yearStr not in curYears:
        print('Creating year folder')
        '''
        All we need to specifiy for the folder
        In [30]: folder_schema = server.get_schema('Folder')
        In [31]: [ x for x in folder_schema if folder_schema[x]['required'] ]
        Out[31]: ['title']
        '''
        #Then try to create the year folder 
        newYearFolder = {baseFolder+yearStr: [{'title': yearStr, 'description': 'Labbook entries for {0}'.format(yearStr)}, 'Folder']}
        ploneServer.post_object(newYearFolder)

    #Check whether we have the month folder and create if necessary
    tmpContent = ploneServer.get_object([baseFolder+yearStr]).values()[0]
    curMonths = [x['Title'] for x in tmpContent[2]['contents'].values()]
    monthStr = folderDate.strftime('%B')
    if monthStr not in curMonths:
        print('Creating month folder')
        newMonthFolder = {baseFolder+yearStr+'/'+monthStr.lower(): [{'title': monthStr, 'description': 'Labbook entries for {0} {1}'.format(monthStr, yearStr)}, 'Folder']}
        ploneServer.post_object(newMonthFolder)
        
    #Check whether we have the day folder and create if necessary
    tmpContent = ploneServer.get_object([baseFolder+yearStr+'/'+monthStr.lower()]).values()[0]
    curDays = [x['Title'] for x in tmpContent[2]['contents'].values()]
    dateStr = folderDate.strftime('%d %B %Y')
    fullPath = baseFolder+yearStr+'/'+monthStr.lower()+'/'+folderDate.strftime('%d-%B-%Y').lower()
    if dateStr not in curDays:
        print('Creating day folder')
        newDayFolder = {fullPath: [{'title': dateStr, 'description': 'Data and notes for the '+ dateStr}, 'Folder']}
        ploneServer.post_object(newDayFolder)
    
    return fullPath
        
    

def upload_image(ploneServer, folderPath, imageFile, imageDescrip='', tags=[]):
    '''
    Helper function that upload the image file.
    '''
    assert os.path.isfile(imageFile), 'Oops! The image file does not exist.'
    imageTitle = os.path.splitext(os.path.split(imageFile)[1])[0]
    imagePost = {folderPath+'/'+imageTitle.lower(): [{'title': imageTitle, 'description': imageDescrip, 'Subject': tags, 'image': Binary(open(imageFile, 'rb').read())}, 'Image']}
    print('Image Post %s' % str(imagePost))
    imagePath = ploneServer.post_object(imagePost)
    urlencodedPath = urllib.quote(imagePath[0])
    print('Create a link to this image in Markdown with:\n' + \
         '[![]({0}/@@images/image/preview)]({0})'.format(urlencodedPath))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageFile', action='store', default=None)    
    parser.add_argument('--username', action='store', default=None)    
    parser.add_argument('--password', action='store', default=None)    
    parser.add_argument('--ploneSite', action='store', default=None)
    parser.add_argument('--imageDescrip', action='store', default=None)
    parser.add_argument('--date', action='store', default=None)
    
    inputs =  parser.parse_args(sys.argv[1:])

    #Parse the date string into a python datetime.date object
    folderDate = time.strptime(inputs.date, '%d-%b-%Y')    
    
    ploneServer = ServerProxy('http://' + inputs.username + ':' + inputs.password + '@' + inputs.ploneSite)

    notebookFolder = '/QLab/lab-notebook/day-to-day/'

    datePath = make_daily_folder(ploneServer, notebookFolder, folderDate)
    upload_image(ploneServer, datePath, inputs.imageFile, inputs.imageDescrip)
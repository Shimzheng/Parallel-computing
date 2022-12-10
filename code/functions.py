"""
Data processing functions to be called in main.py

"""

import json
import numpy as np
import pandas as pd

def read_sydGrid(sydGrid_path):
    """
    param: Sydney grid file path
    return: Sydney grid file
    """

    with open(sydGrid_path, 'r', encoding= 'utf-8') as f:
        grid = json.loads(f.read())
        return grid


def read_langCode(langCode_path):
    """
    param: language code file path
    return: {language_code: language_name} - language code dictionary
    """

    langCode = {}
    with open(langCode_path, 'r', encoding= 'utf-8') as f:
        for line in f:
            (val, key) = line.split()
            langCode[key] = val
    return langCode 


def compute_gridLines(grid):
    """
    param: Sydney grid file
    return: latitude and longitude of the gridlines - sorted lists 
    """

    lat = [] # latitude list
    lng = [] # longitude list
    for d in grid['features']:
        # get the coordinates of the left and upper boundary for each cell
        min_cord = min(d['geometry']['coordinates'][0], key = lambda x: (x[0], -x[1]))
        lat.append(min_cord[-1])
        lng.append(min_cord[0])

        # get the coordinates of the right and lower boundary for each cell
        max_cord = max(d['geometry']['coordinates'][0], key = lambda x: (x[0], -x[1]))
        lat.append(max_cord[-1])
        lng.append(max_cord[0])
    lat = np.unique(sorted(lat))[::-1]
    lng = np.unique(sorted(lng))

    return lat, lng


def chunk(twitter_path, chunksize, filesize):
 
    """
    param: twitter file path
           size of chunk in bytes
           size of the data file in bytes
    """

    with open(twitter_path, 'rb') as f:
        # get current position in file
        chunk_end = f.tell()

        while True:
            chunk_start = chunk_end
            # move from current position to current position + chunk_size
            f.seek(f.tell() + chunksize)
            # read until the next line to not break lines in the middle
            f.readline()
            # get chunk_end as the offset from the beginning of the file in bytes
            chunk_end = f.tell()
            if chunk_end > filesize:
                chunk_end = filesize
            yield chunk_start, chunk_end - chunk_start
            if chunk_end == filesize:
                break


def read_tweets(twitter_path, start, size):
    """
    param: twitter file path
           position to start reading in bytes
           size of data to read in bytes
    return: twitter lines
    """

    with open(twitter_path, 'rb') as f:
        # get current position in file
        f.seek(start)
        end = start

        while True:
            line = f.readline()
            end = f.tell()
            yield line
            if end > start + size:
                end = start + size
            if end == start + size:
                break


def proc_line(line):
    """
    param: tweet lines
    return: processed tweet lines
    """

    line = line.decode('utf-8')
    line = line.strip('\n')
    line = line.strip()
    if line.endswith (','):
        line = line[:-1]
    if line.endswith (']}'): 
        line = line[:-2]

    return line

    
def tweet_info(line):
    """
    param: processed twitter file lines
    return: latitude, longitude and language code of the tweet
    """

    x = 0
    y = 0
    code = 0
    try:
        lines = json.loads(line)
        coordinate = lines['doc']['coordinates']
        lang = lines['doc']['metadata']['iso_language_code']
        
        if coordinate != None and lang != None and lang != 'und':
            x = coordinate['coordinates'][0] # longitude
            y = coordinate['coordinates'][-1] # latitude
            code = lang

        else:
            pass
         
    except Exception as e:
        print("Error reading line --ignoring")
        print(line)
        pass
        
    return x, y, code


def cell_allocation(x, y, lat, lng):
    """
    param: latitude and longitude of a single tweet
           Sorted lists of latitudes and longitudes of the gridlines
    return: cell allocation of the tweet - int
    """
    
    cell = 0
    # check if latitude and longitude are outside the of the grid
    if x < lng[0] or -y < -lat[0] or x > lng[-1] or -y > -lat[-1]:
        pass

    else:
        # using loop to account for cases when grid size isn't known
        for i in range(1, len(lng)):
            # check for columns in the grid
            if x <= lng[i]:
                for j in range(1, len(lat)):
                    #check for rows in the grid
                    if -y < -lat[j]:
                        #find the cell as a number  
                        cell = i + (j-1)*(len(lat)-1)
                        break
                    else:
                        continue
                break
            else:
                continue

    return cell


def tweet_processing(code, cell, cell_count, cell_lang):
    """
    param: language code of the tweet
           cell allocation of the tweet
           {cell : count} cell_count dictionary
           {cell: {language: count}} cell_lang nested dictionary        
    return: {cell: {language: count}} - nested cell, language code, count dictionary
            {cell : count} - cell tweet count dictionary
    """
    
    if code == 'zh-tw' or code == 'zh-cn':
        code = 'zh'
        

    if code == 'in':
        code = 'id'

    # keep track of total number of tweets in cells
    count = cell_count.get(cell, 0) + 1
    cell_count[cell] = count

    # keep track of total number of tweets for per language in cells
    num = cell_lang[cell].get(code, 0) + 1
    cell_lang[cell][code] = num

    return cell_count, cell_lang


def sum_cell_count(gather_cell_count):
    """
    param: a list of {cell : count} cell_count dictionary      
    return: {cell : count} a combined cell_count dictionary
    """

    n = len(gather_cell_count[0].keys())
    final_cell_count = {i:0 for i in range(1, (n+1))}
    for d in gather_cell_count:
        for k in d.keys():
            final_cell_count[k] +=d[k]

    return final_cell_count


def sum_cell_lang(gather_cell_lang):
    """
    param: a list of {cell: {language: count}} cell_lang nested dictionary     
    return: {cell: {language: count}} a combined cell_lang nested dictionary 
    """

    n = len(gather_cell_lang[0].keys())
    final_cell_lang = {i:{} for i in range(1, (n+1))}
    for d in gather_cell_lang:
        for i in range(1, n+1):
            for k in d[i].keys():
                num = final_cell_lang[i].get(k, 0) + d[i][k]
                final_cell_lang[i][k] = num

    return final_cell_lang


def final_results(langCode, final_cell_count, final_cell_lang):
    """
    param: {language_code: language_name} - language code dictionary
           {cell : count} a combined cell_count dictionary
           {cell: {language: count}} a combined cell_lang nested dictionary        
    return: print as a table - 
            cell name
            total tweet count per cell
            total language used per cell
            top-10 language used per cell           
    """

    num2string = {1:'A1', 2 :'A2', 3:'A3', 4:'A4', 5:'B1', 6:'B2', 7:'B3', 8:'B4', 
    9:'C1', 10:'C2', 11:'C3', 12:'C4', 13:'D1', 14:'D2', 15:'D3', 16:'D4'}

    #mapping cell names to cell numbers
    cell = [v2 for k1 in final_cell_count.keys() for k2, v2 in num2string.items() if k1 == k2]
    #get the total number of tweets in cells
    tot_tweet = [v for v in final_cell_count.values()]

    #calculate the number of languages used
    num_lang = [len(v) for v in final_cell_lang.values()]

    for key in final_cell_lang.keys():
        #replace language codes with language names
        final_cell_lang[key] = {v2: v1 for k1, v1 in final_cell_lang[key].items() for k2, v2 in langCode.items() if k1 == k2}
        #sort tweet languages according to frequency & get top-10
        final_cell_lang[key] = sorted(final_cell_lang[key].items(), key = lambda item: item[1])[::-1][:10]
    
    #get the combined mostly tweeted languages
    top_lang = [v for v in final_cell_lang.values()]

    count_result = pd.DataFrame(list(zip(cell, tot_tweet, num_lang)), 
    columns = ['Cell', '#Tweets', '#Number of Languages Used'])

    lang_result = pd.DataFrame(list(zip(cell, top_lang)), 
    columns = ['Cell', '#Top 10 Languages & #Tweets'])

    return print(count_result.to_string(index=False)), print(lang_result.to_string(index=False))
  

#############################################################################################################

# def cell_allocation(x, y, lat, lng):
#     """
#     param: Latitude and longitude of a single tweet
#            Sorted lists of latitudes and longitudes of the gridlines
#     return: Cell allocation of the tweet - int
#     """

#     cell = 0 

#     if x < lng[0] or -y < -lat[0] or x > lng[-1] or -y > -lat[-1]:
#         pass

#     else:

#         diff_x = x - lng
#         ind_x = np.argmin(abs(diff_x))

#         if diff_x[ind_x] > 0:
#             cellx = ind_x + 1
#         else:
#             cellx = ind_x
    
#         diff_y = lat - y
#         ind_y = np.argmin(abs(diff_y))

#         if diff_y[ind_y] >= 0:
#             celly = ind_y + 1
#         else:
#             celly = ind_y
  
#         cell = cellx + (celly-1)*(len(lat)-1)

#     return cell
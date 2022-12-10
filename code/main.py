"""
main

"""

import argparse
import os
from datetime import datetime
from mpi4py import MPI
from functions import *

parser = argparse.ArgumentParser(description = 'CCC-A1')

# data: tiny/small/bigTwitter.json
parser.add_argument('-data', help='path to twitter data file', type=str, default = 'tinyTwitter.json')
# grid: sydGrid.json
parser.add_argument('-grid', help='path to sydGrid file', type=str, default = 'sydGrid.json')
# code: langCode.json
parser.add_argument('-code', help='path to language code and names file', type=str, default = 'langCode.json')

args = parser.parse_args()


def main():
    
    total_start = datetime.now()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        langCode = read_langCode(args.code) # load langCode data

        sydGrid_data = read_sydGrid(args.grid) # load grid data 
        lat, lng = compute_gridLines(sydGrid_data)

        filesize = os.path.getsize(args.data) # os read size of the datafile
        chunksize = filesize // size # calculate byte size to be processed by each process 
        
        chunks = []
        for chunkStart, chunkSize in chunk(args.data, chunksize, filesize):
            chunks.append({'chunkStart': chunkStart, 'chunkSize': chunkSize})

    else:
        lat = None
        lng = None
        chunks = None

    #################################################################################################
    ########################## Parallel reading and processing twitter data #########################
    #################################################################################################
    
    # Boarcast sends latitude and longitude of the grid to each individual process
    lat_in_allrank = comm.bcast(lat, root=0)
    lng_in_allrank = comm.bcast(lng, root=0)

    # Scatter sends chunk to each individual process (i.e. chunkStart, chunkSize)
    chunk_per_process = comm.scatter(chunks, root=0)
    
    ###################################################################################################
    ############################ sanity check for testing  - muted ####################################
    #
    # print('Rank ' + str(rank) + ' received chunk - chunkStart: ' + str(
    #     chunk_per_process['chunkStart']) + ' -  chunkSize ' +
    #       str(chunk_per_process['chunkSize']))
    ###################################################################################################

    comm.Barrier() # wait until all processes are ready

    # proc_start = datetime.now()

    # allow for cases when size of grid is unknown
    cell_N = (len(lat_in_allrank) - 1) * (len(lng_in_allrank) - 1)
    cell_lang = {i:{} for i in range(1, cell_N + 1)}
    cell_count = {i: 0 for i in range(1, cell_N + 1)}

    yield_tweets = read_tweets(args.data, chunk_per_process['chunkStart'], chunk_per_process['chunkSize'])

    for line in yield_tweets:
        line_processed = proc_line(line)
        x, y, code = tweet_info(line_processed) 
        if x != 0 and y != 0:
            cell = cell_allocation(x, y, lat_in_allrank, lng_in_allrank) 
            if cell != 0:
                cell_count, cell_lang = tweet_processing(code, cell, cell_count, cell_lang) 


    # proc_end = datetime.now()
    # print(f'I am rank {rank}, my data processing time is {proc_end - proc_start}')    

    if size > 1: # when there is more than one process

        gather_cell_count = comm.gather(cell_count, root = 0)
        gather_cell_lang = comm.gather(cell_lang, root = 0)

        comm.Barrier()

        # gather_end = datetime.now()
        # print(f'I am rank {rank}, my gathering time is {gather_end - proc_end}')
        
        if rank == 0:
            
            # result_start = datetime.now()
            final_cell_count = sum_cell_count(gather_cell_count)
            final_cell_lang = sum_cell_lang(gather_cell_lang)
            # result_end = datetime.now()
            # print(f'I am rank {rank}, results combining time = {result_end - result_start}')

    else:
        
        final_cell_count = cell_count
        final_cell_lang = cell_lang
    
    if rank == 0:

        # result_start = datetime.now()
        final_results(langCode, final_cell_count, final_cell_lang)
        # result_end = datetime.now()
        # print(f'I am rank {rank}, results preparing time = {result_end - result_start}')
            
        total_end = datetime.now()
        print(f'Total processing time = {total_end - total_start}')


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
'''
    ** CURRENT VERSION -- version 0.5 **
    Notes: 
    - passing datalog/yaml file between functions is efficient
    - no arg parser yet for datalog file --> currently copy/pasting filename
    - z_anode =/= z-coordinate
    - yaml file == y == tile layout (make sure it's the correct one)
    - I have no idea what unassoc_hits and unassoc_hit_mask are
            (what are they supposed to be (unassoc)iated with?)
    - event_dictionary contains all info on events & hits 
'''

import h5py as h
import numpy as np
import yaml

'''
    import datalog file and return==> change datalog filename here
    called by: init
'''
def import_data_file():
    data_file = 'datalog_2021_04_01_22_07_34_CESTevd.h5'
    #data_file = 'datalog_2021_04_01_11_26_21_CESTevd.h5' ==> file has incorrect data logging

    try:
        f = h.File(data_file, 'r')
    except FileNotFoundError:
        print('couldn\'t import {}'.format(data_file))
        
    print('file {} import successful'.format(data_file))
    return f


'''
    import yaml file and return
    called by: init
'''
def import_yaml_file():
    yaml_file = 'multi_tile_layout-2.1.16.yaml'
    
    try:
        with open(yaml_file, 'r') as _file:
            y = yaml.load(_file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print('could\'nt import {}'.format(yaml_file))

    print('{} import successful'.format(yaml_file))
    return y
   

'''
    Obtaining events that satisfy criteria of: 
        - # of hits ==> f['events'][i][3]
        - q         ==> f['events'][i][4]
        - ts_start  ==> f['events'][i][5] (to eliminate initial start up events) 
    ==> definitely needs to change
    called by: driver
'''
def obtain_events(f):
    events        = f['events']
    hits          = f['hits']
    stored_events = []
    count         = 0

    for i in range(len(f['events'])):
        if (f['events'][i][3] > 1e4 and f['events'][i][4] > 1e3 and f['events'][i][5] != 0):
            print('Event {} has {} hits'.format(events[i][0], events[i][3]))
            print('     timestamps: start = {}, end = {}'.format(events[i][5], events[i][6])) 
            print('     q = {}'.format(events[i][4]))
            stored_events.append(events[i][0])
            count += 1
        else:
            continue
    print('{} events stored'.format(count))
    return stored_events 

'''
    Outputs individual z-coordinate per hit
    called by: driver
    - unassoc_hit_time := respective masked time of hit 
'''
def obtain_z_coordinate(tile_id, tile_positions, tile_orientations, f_info, time):
    z_anode         = tile_positions[tile_id - 1][0]        # i
    drift_direction = tile_orientations[tile_id - 1][0]     # f
    vdrift          = f_info['vdrift']                      # f
    clock_period    = f_info['clock_period']                # f
   
    z_coordinate    = z_anode + (time * vdrift * clock_period * drift_direction)

    return z_coordinate
    

'''
    Returns the start time of an event
    ==> c/p'd from module0_evd.py with some modifications
    called by: get_z_coordinates
'''
def obtain_event_start_time(event, _hits):
    ticks_per_qsum      = 10 # clock ticks per time bin
    t0_charge_threshold = 200.0 # Rough qsum threshold
    hit_ref             = event['hit_ref']
    hits                = _hits[hit_ref]
    
    # Determine charge vs time in enlarged window
    # - if event is long enough, calculate qsum vs time
    min_ts = np.amin(hits['ts'])
    max_ts = np.amax(hits['ts'])
    if (max_ts - min_ts) > ticks_per_qsum:
        time_bins = np.arange(min_ts - ticks_per_qsum,
                                max_ts + ticks_per_qsum)
        # - integrate q in sliding window to produce qsum profile
        # - histogram raw charge
        q_vs_t = np.histogram(hits['ts'],
                                bins=time_bins,
                                weights=hits['q'])[0]
        
        #  calculate rolling qsum
        qsum_vs_t = np.convolve(q_vs_t,
                                np.ones(ticks_per_qsum,dtype=int),
                                'valid')
        t0_bin_index = np.argmax(qsum_vs_t > t0_charge_threshold)
        t0_bin_index += ticks_per_qsum
        start_time = time_bins[t0_bin_index]
        # Check if qsum exceed threshold
        if start_time < max_ts:
            return start_time
    
    # Fallback is to use the first hit
    return event['ts_start']
    

'''
    Obtains all groupings of io_group and io_channel needed in order to find tile_id's
    - c/p'd from module0_evd.py with some modifications 
    called by: driver
'''
def get_io_group_io_channel_to_tile(y):
    mm2cm                       = 0.1
    pixel_pitch                 = y['pixel_pitch'] * mm2cm
    chip_channel_to_position    = y['chip_channel_to_position']
    tile_chip_to_io             = y['tile_chip_to_io']
    io_group_io_channel_to_tile = {}
    
    for tile in tile_chip_to_io:
        for chip in tile_chip_to_io[tile]:
            io_group_io_channel  = tile_chip_to_io[tile][chip]
            io_group             = io_group_io_channel//1000
            io_channel           = io_group_io_channel%1000
            
            io_group_io_channel_to_tile[(io_group,io_channel)] = tile
    
    return io_group_io_channel_to_tile


'''
    Returns tile identification based on io_group and io_channel from individual hit
    according to io_group_io_channel_to_tile specification
    called by: driver
'''
def get_tile_id(io_group_io_channel_to_tile, io_group, io_channel):
    try:
        tile_id = io_group_io_channel_to_tile[io_group, io_channel]
    except:
        print("io group %i, io channel %i not found" % (io_group, io_channel))

    return tile_id


'''
    Main driver for finding necessary hit information from events
    called by: init
'''
def driver(f, y):
    # module0_evd method of obtaining positions/orientations of tiles,
    # and finding associated tile
    tile_positions              = np.array(list(y['tile_positions'].values()))
    tile_orientations           = np.array(list(y['tile_orientations'].values()))
    io_group_io_channel_to_tile = get_io_group_io_channel_to_tile(y)

    _events           = f['events']                   # unbuffered event list
    events            = _events[_events['nhit'] > 1]  # buffered
    hits              = f['hits']                     # all hits
    tracks            = f['tracks']                   # all tracks
    f_info            = f['info'].attrs               # attributes of datafile
    event_dictionary  = {}                            # dictionary containing all event/hit info
    event_start_time  = 0                             # initialize
    
    print('obtaining events based on selection criteria')
    stored_events     = obtain_events(f)              # obtain events based on selection criteria
    print('finished obtaining events')
    print('iterating through all stored events')
    # Iterate through events with criteria
    for i in range(len(stored_events)):
        ev_id         = stored_events[i]        # id of an event
        event         = events[ev_id]           # individual event
        hit_ref       = event['hit_ref']        # specifies hits associated with event
        specific_hits = hits[hit_ref]           # hits specific to an event
    
        unassoc_hit_mask  = np.ones(event['nhit']).astype(bool)     # mask implemented on event
        unassoc_hits      = specific_hits[unassoc_hit_mask]         # list of unassoc hits (uh)
                                                                    # uh[0]    ==> first hit
                                                                    # uh[0][3] ==> timestamp
        
        event_start_time  = obtain_event_start_time(event, hits)    # timestamp of event beginning
        hit_dictionary    = {}                                      # will contain information about a hit
        # Iterate through an event's hits and obtain information needed to get z-coordinates
        for j in range(len(specific_hits)):
            _io_group     = specific_hits[j][6]
            _io_channel   = specific_hits[j][5]
            _tile_id      = get_tile_id(io_group_io_channel_to_tile, _io_group, _io_channel)
            _timestamp    = unassoc_hits[j][3] - event_start_time
            _z_coordinate = obtain_z_coordinate(_tile_id, tile_positions, tile_orientations, f_info, _timestamp)
            _q  = specific_hits[j][4]
            _px = specific_hits[j][1]
            _py = specific_hits[j][2]

            # each hit has its own dictionary, containing:
            # -- [tile id, q, timestamp, px, py, z]
            hit_dictionary[specific_hits[j][0]] = [_tile_id, _q, _timestamp, _px, _py, _z_coordinate]
        
        '''
            Store all hits from an event in the dictionary as : 
            {evid_1:{hid_1:[tile_id_1, q_1, time_1, px_1, py_1, pz_1],
                   hid_2:[tile_id_2, q_2, time_2, px_2, py_2, pz_2], 
                   ...},
            evid_2:{hid_1:[tile_id_1, q_1, time_1, px_1, py_1, pz_1],
                   hid_2:[tile_id_2, q_2, time_2, px_2, py_2, pz_2], 
                   ...},
            ...  
             }
        '''
        event_dictionary[stored_events[i]] = hit_dictionary
        print('event {} information stored'.format(stored_events[i]))

    # default printing to verify tagged events, but can print everything else
    print('event_dictionary keys: {}'.format(event_dictionary.keys()))


'''
   test/debugging function
'''
def test_function(f):                                     
    print('in test function')                             


'''
    main init function
'''
if __name__ == '__main__':                                
    f = import_data_file()  # evd data file                    
    y = import_yaml_file()  # yaml tile layout file               
    driver(f, y)                    
    print('done')

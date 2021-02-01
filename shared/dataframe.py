import pandas as pd
import numpy as np

"""
# ---------------------------------------------------------------------------------------------------
# FUNCTIONS for PD.DATAFRAME/LIST/UMANAGER/NUMBER
# ---------------------------------------------------------------------------------------------------

pd.DataFrame related:

    add_columns
        FUNCTION: add multiple columns to an existed pd.DataFrame
        SYNTAX:   add_columns(dataframe: pd.DataFrame, name_lst: list, value_lst: list)
    
    pd_numeric
        FUNCTION: numeric certain columns within a pd.DataFrame
        SYNTAX:   pd_numeric(dataframe: pd.DataFrame, column_lst: list)
    
    copy_based_on_index
        FUNCTION: copy data from dataframe2 to dataframe1 based on common index
        SYNTAX:   copy_based_on_index(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame,
                  index_name1: str, index_name2: str, column_lst1: list, column_lst2: list)
                  
List related:

    find_pos
        FUNCTION: find the position of the first value in linearly increased list that is larger 
                  than or equal to the given value
        SYNTAX:   find_pos(value: int or float, increase_lst: list)
    
    list_subtraction
        FUNCTION: subtract elements in two lists in an element wise manner
        SYNTAX:   list_subtraction(list1: list, list2: list)

uManager related:

    get_time_tseries
        FUNCTION: get acquisition time from uManager data
        SYNTAX:   get_time_tseries(store, cb)
    
    get_time
        FUNCTION: transform time information from uManager metadata format into hour, minute and 
                  second
        SYNTAX:   get_time(frame: int, time_tseries: list)
    
    get_time_length
        FUNCTION: calculate time between two frames (unit: second)
        SYNTAX:   get_time_length(start_frame: int, end_frame: int, time_tseries: list)
    
    get_frame
        FUNCTION: get action frame number based on metadata provided time
        SYNTAX:   get_frame(time_lst: list, acquire_time_tseries: list)
    
    get_pixels_tseries
        FUNCTION: get pixels time series
        SYNTAX:   get_pixels_tseries(store, cb, data_c: int)
        
Number related:

    get_grid_pos
        FUNCTION: transform sequential positions into column/row positions in a grid (n x n snake 
                  manner)
        SYNTAX:   get_grid_pos(pos: int, num_grid: int)
          
    find_closest
        FUNCTION: find closest spot
        SYNTAX:   find_closest(aim_x: int or float, aim_y: int or float, x_list: list, y_list: list)
  
"""

# ---------------------------------------------------------------------------------------------------
# FUNCTIONS for PD.DATAFRAME
# ---------------------------------------------------------------------------------------------------


def add_columns(dataframe: pd.DataFrame, name_lst: list, value_lst: list):
    """
    Add multiple columns to an existed pd.DataFrame

    :param dataframe: pd.DataFrame
    :param name_lst: list, list of column names
    :param value_lst: list, list of corresponding values
    :return: dataframe: pd.DataFrame with columns added

    """
    for i in range(len(name_lst)):
        dataframe[name_lst[i]] = value_lst[i]

    return dataframe


def pd_numeric(dataframe: pd.DataFrame, column_lst: list):
    """
    Numeric certain columns within a pd.DataFrame

    :param dataframe: pd.DataFrame
    :param column_lst: name of columns that need to be numeric
    :return: dataframe: numeric pd.DataFrame

    """
    for x in column_lst:
        dataframe[x] = pd.to_numeric(dataframe[x], errors='coerce')

    return dataframe


def copy_based_on_index(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame,
                        index_name1: str, index_name2: str,
                        column_lst1: list, column_lst2: list):
    """
    Copy data from dataframe2 to dataframe1 based on common index

    :param dataframe1: pd.DataFrame
                dataframe that receives information
    :param dataframe2: pd.DataFrame
                dataframe that provides information
    :param index_name1: str
                common index column name in dataframe1
    :param index_name2: str
                common index column name in dataframe2
    :param column_lst1: list
                list of column names assigned to dataframe1 when add new columns that copy
                from dataframe2, len(column_lst1) = len(column_lst2)
    :param column_lst2: list
                list of column names in dataframe2 that ready for copy, , len(column_lst1)
                = len(column_lst2)
    :return: dataframe1: pd.DataFrame, dataframe1 with new columns added

    """

    value_lst = [[] for _ in range(len(column_lst1))]

    for i in dataframe1[index_name1]:
        target = dataframe2[dataframe2[index_name2] == i].iloc[0]
        for j in range(len(column_lst1)):
            value_lst[j].append(target[column_lst2[j]])
    dataframe1 = add_columns(dataframe1, column_lst1, value_lst)

    return dataframe1


# replace values in some rows with other values in a dataframe
# log_pd.loc[log_pd.x == 0, 'x'] = log_pd[log_pd['x'] == 0]['aim_x']
# log_pd.loc[log_pd.y == 0, 'y'] = log_pd[log_pd['y'] == 0]['aim_y']

# ---------------------------------------------------------------------------------------------------
# FUNCTIONS for LIST
# ---------------------------------------------------------------------------------------------------


def find_pos(value: int or float, increase_lst: list):
    """
    Find the position of the first value in linearly increased list that is larger than or equal to the
    given value

    Note: if value > max(increase_lst), which means such position does not exist in the given list. This
        function will return len(increase_lst), which is the last position of the list + 1

    Usage examples:
    1) used to look for bleach frame
       > bleach_frame = find_pos(bleach_time, time_tseries)

    :param value: int or float
    :param increase_lst: list, has to be a linearly increased list
    :return: out: position of the first value in linearly increased list that is larger than or equal to
                the given value, start from 0

    """
    out = len(increase_lst)
    i = 0
    while i < len(increase_lst):
        if value <= increase_lst[i]:
            out = i
            break
        else:
            i += 1

    return out


def list_subtraction(list1: list, list2: list):
    """
    Subtract elements in two lists in an element wise manner

    Usage examples:
    1) > list_subtracted = list_subtraction([2, 3, 4], [1, 1, 5])
       > print(list_subtracted)
       > [1, 2, -1]

    :param list1: list
                list1[i]: float or int
    :param list2: list
                list2[i]: float or int
    :return: list_subtracted: list
                list_subtracted[i]: float or int

    """
    if len(list1) == len(list2):
        list_subtracted = [list1_i - list2_i for list1_i, list2_i in zip(list1, list2)]
    else:
        raise ValueError("length of two provided lists for subtraction do not match.")

    return list_subtracted

# ---------------------------------------------------------------------------------------------------
# FUNCTIONS for UMANAGER
# ---------------------------------------------------------------------------------------------------


def get_time_tseries(store, cb):
    """
    Get acquisition time from uManager data

    Usage examples:
    1) find bleach frame for FRAP analysis

    :param store: store = mm.data().load_data(data_path, True)
    :param cb: cb = mm.data().get_coords_builder()
    :return: acquire_time_tseries: list
                list of acquisition time displayed in 'hour:min:sec' format
                e.g. ['17:30:38.360', '17:30:38.455', '17:30:38.536', '17:30:38.615', ...]
             real_time_tseries: list
                list of time in sec with first frame set as 0
                e.g. [0.0, 0.09499999999999886, 0.17600000000000193, 0.25500000000000256, ...]

    """
    # get acquisition time
    acquire_time_tseries = []

    max_t = store.get_max_indices().get_t()
    for t in range(0, max_t):
        img = store.get_image(cb.t(t).build())
        acq_time = img.get_metadata().get_received_time().split(' ')[1]
        acquire_time_tseries.append(acq_time)

    real_time_tseries = []
    for i in range(max_t):
        real_time_tseries.append(get_time_length(0, i, acquire_time_tseries))

    return acquire_time_tseries, real_time_tseries


def get_time(frame: int, time_tseries: list):
    """
    Transform time information from uManager metadata format into hour, minute and second

    :param frame: int, given frame
    :param time_tseries: list
                list of time displayed in 'hour:min:sec' format
                e.g. ['17:30:38.360', '17:30:38.455', '17:30:38.536', '17:30:38.615', ...]
    :return: t_hr, t_min, t_sec: time of given frame displayed in hour, minute and second

    """
    t_hr = int(time_tseries[frame].split(':')[0])
    t_min = int(time_tseries[frame].split(':')[1])
    t_sec = float(time_tseries[frame].split(':')[2])

    return t_hr, t_min, t_sec


def get_time_length(start_frame: int, end_frame: int, time_tseries: list):
    """
    Calculate time between two frames (unit: second)

    :param start_frame: int, start frame
    :param end_frame: int, end frame
    :param time_tseries: list
                list of time displayed in 'hour:min:sec' format
                e.g. ['17:30:38.360', '17:30:38.455', '17:30:38.536', '17:30:38.615', ...]
    :return: out: time difference between two frames in second
    """
    t_start = get_time(start_frame, time_tseries)
    t_end = get_time(end_frame, time_tseries)

    out = 3600*(t_end[0]-t_start[0]) + 60*(t_end[1]-t_start[1]) + (t_end[2]-t_start[2])

    return out


def get_frame(time_lst: list, acquire_time_tseries: list):
    """
    Get action frame number based on metadata provided time

    Usage examples:
    1) get bleach frame for FRAP analysis

    :param time_lst: list
                list of action time displayed in metadata format
    :param acquire_time_tseries: list
                list of acquisition time displayed in 'hour:min:sec' format
                e.g. ['17:30:38.360', '17:30:38.455', '17:30:38.536', '17:30:38.615', ...]
    :return: frame_lst: list, list of action frames

    """
    frame_lst = []  # frame number of or right after photobleaching
    for i in range(len(time_lst)):
        # number of first frame after photobleaching (num_pre)
        num_pre = find_pos(time_lst[i].split(' ')[1], acquire_time_tseries)
        frame_lst.append(num_pre)

    return frame_lst


def get_pixels_tseries(store, cb, data_c: int):
    """
    Get pixels time series

    :param store: store = mm.data().load_data(data_path, True)
    :param cb: cb = mm.data().get_coords_builder()
    :param data_c: channel to be analyzed
    :return: pixels_tseries: list, pixels time series
                e.g. [pixels_t0, pixels_t1, pixels_t2, ...]

    """
    max_t = store.get_max_indices().get_t()
    pixels_tseries = []
    for t in range(0, max_t):
        img = store.get_image(cb.z(0).p(0).c(data_c).t(t).build())
        pixels = np.reshape(img.get_raw_pixels(), newshape=[img.get_height(), img.get_width()])
        pixels_tseries.append(pixels)

    return pixels_tseries


# ---------------------------------------------------------------------------------------------------
# FUNCTIONS for NUMBER
# ---------------------------------------------------------------------------------------------------


def get_grid_pos(pos: int, num_grid: int):
    """
    Transform sequential positions into column/row positions in a grid (n x n snake manner)

    Usage example:
    1) > print(get_grid_pos(3, 4))
       > [0, 2]
       > print(get_grid_pos(7, 4))
       > [1, 0]

    :param pos: sequential positions
    :param num_grid: size of the grid
    :return: row, col: row/column positions

    """
    row = pos//num_grid
    if row % 2 == 0:
        col = pos-row*num_grid
    else:
        col = num_grid-1-(pos-row*num_grid)

    return row, col


def find_closest(aim_x: int or float, aim_y: int or float, x_list: list, y_list: list):
    """
    Find closest spot

    :param aim_x: x coordinate of aim
    :param aim_y: y coordinate of aim
    :param x_list: list of x coordinates
    :param y_list: list of y coordinate
    :return: x_closest, y_closest: x, y coordinates of the closest spot

    """
    dis = 10000000000
    x_closest = 0
    y_closest = 0
    for i in range(len(x_list)):
        x_temp = x_list[i]
        y_temp = y_list[i]
        dis_temp = (x_temp - aim_x) ** 2 + (y_temp - aim_y) ** 2
        if dis_temp < dis:
            dis = dis_temp
            x_closest = x_temp
            y_closest = y_temp

    return x_closest, y_closest

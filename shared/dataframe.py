import pandas as pd

"""
# ---------------------------------------------------------------------------------------------------
# FUNCTIONS for PD.DATAFRAME/LIST/UMANAGER/NUMBER
# ---------------------------------------------------------------------------------------------------

pd.DataFrame related:

    add columns
        FUNCTION: add multiple columns to an existed pd.DataFrame
        SYNTAX:   add_columns(dataframe: pd.DataFrame, name_lst: list, value_lst: list)

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

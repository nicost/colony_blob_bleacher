# -----------------------------------
# FUNCTIONS for PD.DATAFRAME
# -----------------------------------

def add_columns(pd, name: list, value: list):
    """
    Add multiple columns to a pandas dataframe.

    :param pd: pandas.dataframe
    :param name: list of column names
    :param value: list of corresponding values
    :return: pd: pandas.dataframe with columns added
    """
    for i in range(len(name)):
        pd[name[i]] = value[i]

    return pd


def find_pos(val, increase_list: list):
    """
    Return the position of the first value in linearly increased list that is larger than or equal to val.

    :param val: bleach time to be tested
    :param increase_list: list of acquisition times of each frame
    :return: out: first frame after or during photobleaching
    """
    out = -1
    i = 0
    while i < len(increase_list):
        if val <= increase_list[i]:
            out = i
            break
        else:
            i += 1

    return out


def add_object_measurements(pd, name, obj_name, value):
    """
    Add measurements from object into corresponding pandas dataframe.

    :param pd: pd.DataFrame
    :param name: added column name
    :param obj_name: name of the objects where measurement made from
    :param value: list of measurements
    :return:
    """
    pd_sort = pd.sort_values(by=obj_name).reset_index(drop=True)
    pd_sort[name] = value

    return pd_sort


def get_time(t_frame, t_time):
    """
    extract time information from metadata time info
    :param t_frame: given frame
    :param t_time: acquisition time from metadata info
    :return: time in hour, minute and second
    """
    t_hr = int(t_time[t_frame].split(':')[0])
    t_min = int(t_time[t_frame].split(':')[1])
    t_sec = float(t_time[t_frame].split(':')[2])

    return t_hr, t_min, t_sec


def get_time_length(t_start_frame, t_end_frame, t_time):
    """
    get the time length between two frames (unit: second)
    :param t_start_frame: start frame
    :param t_end_frame: end frame
    :param t_time: acquisition time from metadata info
    :return: time difference between two frames
    """
    t_start = get_time(t_start_frame, t_time)
    t_end = get_time(t_end_frame, t_time)

    out = 3600*(t_end[0]-t_start[0]) + 60*(t_end[1]-t_start[1]) + (t_end[2]-t_start[2])

    return out

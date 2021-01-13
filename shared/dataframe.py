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

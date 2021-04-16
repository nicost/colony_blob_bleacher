import os
import pandas as pd
import shared.display as dis

multi_data_source = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/"\
    "20210408_CBB_nucleoliFRAPscreen1/dataFiles/"
WT_source = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/"\
    "20210408_CBB_nucleoliFRAPscreen1/WTFiles/"
save_source = "/Users/xiaoweiyan/Dropbox/LAB/ValeLab/Projects/Blob_bleacher/Data/"\
    "20210408_CBB_nucleoliFRAPscreen1/dataSummary1/"
ctrl_lst = ['G8', 'G9', 'G10', 'G11']

analyze_organelle = 'nucleoli'  # only accepts 'sg' or 'nucleoli'
analysis_mode = 'single_exp'

inc = 5
limit_frap = 250
limit_organelle = 500
repeat = 50

multi_dirs = [x for x in os.listdir(multi_data_source)]
if '.DS_Store' in multi_dirs:
    multi_dirs.remove('.DS_Store')

# FRAP analysis
data_original_WT = pd.read_csv(("%sWT_data_full.txt" % WT_source), na_values=['.'], sep='\t')
data_original_WT_ft = data_original_WT[data_original_WT['frap_filter_%s' % analysis_mode] == 1]
mob = data_original_WT_ft['%s_mobile_fraction' % analysis_mode].tolist()
curve_mob = data_original_WT_ft['mobile_fraction'].tolist()
t_half = data_original_WT_ft['%s_t_half' % analysis_mode].tolist()
curve_t_half = data_original_WT_ft['t_half'].tolist()
slope = data_original_WT_ft['%s_slope' % analysis_mode].tolist()
curve_slope = data_original_WT_ft['linear_slope'].tolist()
sample = ['WT'] * len(data_original_WT_ft)

# Organelle analysis
if analyze_organelle == 'nucleoli':
    data_original_WT_organelle = pd.read_csv(("%sWT_data_nucleoli.txt" % WT_source), na_values=['.'], sep='\t')
elif analyze_organelle == 'sg':
    data_original_WT_organelle = pd.read_csv(("%sWT_data_sg.txt" % WT_source), na_values=['.'], sep='\t')
size = data_original_WT_organelle['size'].tolist()
raw_int = data_original_WT_organelle['raw_int'].tolist()
circ = data_original_WT_organelle['circ'].tolist()
ecce = data_original_WT_organelle['eccentricity'].tolist()
sample_organelle = ['WT'] * len(data_original_WT_organelle)

# organize data into one dataframe
for i in ctrl_lst:
    data_temp = pd.read_csv(("%s%s/%s_data_full.txt" % (multi_data_source, i, i)), na_values=['.'], sep='\t')
    data_temp_ft = data_temp[data_temp['frap_filter_%s' % analysis_mode] == 1]
    mob = mob + data_temp_ft['%s_mobile_fraction' % analysis_mode].tolist()
    curve_mob = curve_mob + data_temp_ft['mobile_fraction'].tolist()
    t_half = t_half + data_temp_ft['%s_t_half' % analysis_mode].tolist()
    curve_t_half = curve_t_half + data_temp_ft['t_half'].tolist()
    slope = slope + data_temp_ft['%s_slope' % analysis_mode].tolist()
    curve_slope = curve_slope + data_temp_ft['linear_slope'].tolist()
    sample = sample + [i] * len(data_temp_ft)

    if analyze_organelle == 'nucleoli':
        data_organelle_temp = pd.read_csv(("%s%s/%s_data_nucleoli.txt" % (multi_data_source, i, i)),
                                          na_values=['.'], sep='\t')
    elif analyze_organelle == 'sg':
        data_organelle_temp = pd.read_csv(("%s%s/%s_data_sg.txt" % (multi_data_source, i, i)),
                                          na_values=['.'], sep='\t')
    size = size + data_organelle_temp['size'].tolist()
    raw_int = raw_int + data_organelle_temp['raw_int'].tolist()
    circ = circ + data_organelle_temp['circ'].tolist()
    ecce = ecce + data_organelle_temp['eccentricity'].tolist()
    sample_organelle = sample_organelle + [i] * len(data_organelle_temp)

data_name = []
data_frap_n_curve = []
data_frap_phenotype_limit = []
data_frap_phenotype_mob = []
data_frap_phenotype_curve_mob = []
data_frap_phenotype_t_half = []
data_frap_phenotype_curve_t_half = []
data_frap_phenotype_slope = []
data_frap_phenotype_curve_slope = []

data_organelle_n = []
data_organelle_phenotype_limit = []
data_organelle_phenotype_size = []
data_organelle_phenotype_int = []
data_organelle_phenotype_limit_circ = []
data_organelle_phenotype_circ = []
data_organelle_phenotype_ecce = []

for r in range(len(multi_dirs)):
    name = multi_dirs[r]
    print('# Calculating %s ... (%d/%d)' % (name, r+1, len(multi_dirs)))
    data_name.append(name)
    data_source = ("%s%s/" % (multi_data_source, name))
    save_path = ("%s%s/" % (save_source, name))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # organize data into one dataframe
    data_temp = pd.read_csv(("%s%s/%s_data_full.txt" % (multi_data_source, name, name)), na_values=['.'], sep='\t')
    data_temp_ft = data_temp[data_temp['frap_filter_%s' % analysis_mode] == 1]
    mob_sample = mob + data_temp_ft['%s_mobile_fraction' % analysis_mode].tolist()
    curve_mob_sample = curve_mob + data_temp_ft['mobile_fraction'].tolist()
    t_half_sample = t_half + data_temp_ft['%s_t_half' % analysis_mode].tolist()
    curve_t_half_sample = curve_t_half + data_temp_ft['t_half'].tolist()
    slope_sample = slope + data_temp_ft['%s_slope' % analysis_mode].tolist()
    curve_slope_sample = curve_slope + data_temp_ft['linear_slope'].tolist()
    sample_sample = sample + [name] * len(data_temp_ft)

    if analyze_organelle == 'nucleoli':
        data_organelle_temp = pd.read_csv(("%s%s/%s_data_nucleoli.txt" % (multi_data_source, name, name)),
                                          na_values=['.'], sep='\t')
    elif analyze_organelle == 'sg':
        data_organelle_temp = pd.read_csv(("%s%s/%s_data_sg.txt" % (multi_data_source, name, name)),
                                          na_values=['.'], sep='\t')
    size_sample = size + data_organelle_temp['size'].tolist()
    raw_int_sample = raw_int + data_organelle_temp['raw_int'].tolist()
    circ_sample = circ + data_organelle_temp['circ'].tolist()
    ecce_sample = ecce + data_organelle_temp['eccentricity'].tolist()
    sample_organelle_sample = sample_organelle + [name] * len(data_organelle_temp)

    data = pd.DataFrame({'mob': mob_sample, 'curve_mob': curve_mob_sample, 't_half': t_half_sample,
                         'curve_t_half': curve_t_half_sample, 'slope': slope_sample, 'curve_slope': curve_slope_sample,
                         'sample': sample_sample})
    data_organelle = pd.DataFrame({'size': size_sample, 'raw_int': raw_int_sample, 'circ': circ_sample,
                                   'ecce': ecce_sample, 'sample': sample_organelle_sample})

    # save important sample information into lists
    data_sample = data[data['sample'] == name]
    data_WT = data[data['sample'] == 'WT']
    data_frap_n_curve.append(len(data_sample))
    data_frap_phenotype_limit.append(dis.get_phenotype(data_WT, data_sample, 'mob', limit_frap, repeat)[0])
    data_frap_phenotype_mob.append(dis.get_phenotype(data_WT, data_sample, 'mob', limit_frap, repeat)[1])
    data_frap_phenotype_curve_mob.append(dis.get_phenotype(data_WT, data_sample, 'curve_mob', limit_frap, repeat)[1])
    data_frap_phenotype_t_half.append(dis.get_phenotype(data_WT, data_sample, 't_half', limit_frap, repeat)[1])
    data_frap_phenotype_curve_t_half.append(dis.get_phenotype(data_WT, data_sample, 'curve_t_half',
                                                              limit_frap, repeat)[1])
    data_frap_phenotype_slope.append(dis.get_phenotype(data_WT, data_sample, 'slope', limit_frap, repeat)[1])
    data_frap_phenotype_curve_slope.append(dis.get_phenotype(data_WT, data_sample, 'curve_slope',
                                                             limit_frap, repeat)[1])

    data_sample_organelle = data_organelle[data_organelle['sample'] == name]
    data_WT_organelle = data_organelle[data_organelle['sample'] == 'WT']
    data_organelle_n.append(len(data_sample_organelle))
    data_organelle_phenotype_limit.append(dis.get_phenotype(data_WT_organelle, data_sample_organelle, 'size',
                                                            limit_organelle, repeat)[0])
    data_organelle_phenotype_size.append(dis.get_phenotype(data_WT_organelle, data_sample_organelle, 'size',
                                                           limit_organelle, repeat)[1])
    data_organelle_phenotype_int.append(dis.get_phenotype(data_WT_organelle, data_sample_organelle, 'raw_int',
                                                          limit_organelle, repeat)[1])
    data_organelle_phenotype_limit_circ.append(dis.get_phenotype(data_WT_organelle, data_sample_organelle, 'circ',
                                                                 limit_organelle, repeat)[0])
    data_organelle_phenotype_circ.append(dis.get_phenotype(data_WT_organelle, data_sample_organelle, 'circ',
                                                           limit_organelle, repeat)[1])
    data_organelle_phenotype_ecce.append(dis.get_phenotype(data_WT_organelle, data_sample_organelle, 'ecce',
                                                           limit_organelle, repeat)[1])

    # export images
    print("# Export mobile_fraction -ln(p) ...")
    dis.plot_minus_ln_p(inc, limit_frap, repeat, 'mob', data, ctrl_lst, name, save_path)
    dis.plot_minus_ln_p(inc, limit_frap, repeat, 'curve_mob', data, ctrl_lst, name, save_path)
    print("# Export t_half -ln(p) ...")
    dis.plot_minus_ln_p(inc, limit_frap, repeat, 't_half', data, ctrl_lst, name, save_path)
    dis.plot_minus_ln_p(inc, limit_frap, repeat, 'curve_t_half', data, ctrl_lst, name, save_path)
    print("# Export slope -ln(p) ...")
    dis.plot_minus_ln_p(inc, limit_frap, repeat, 'slope', data, ctrl_lst, name, save_path)
    dis.plot_minus_ln_p(inc, limit_frap, repeat, 'curve_slope', data, ctrl_lst, name, save_path)
    print("# Export FRAP violin plots ...")
    dis.plot_violin('mob', data, save_path, name)
    dis.plot_violin('curve_mob', data, save_path, name)
    dis.plot_violin('t_half', data, save_path, name)
    dis.plot_violin('curve_t_half', data, save_path, name)
    dis.plot_violin('slope', data, save_path, name)
    dis.plot_violin('curve_slope', data, save_path, name)

    print("# Export organelle -ln(p) ...")
    dis.plot_minus_ln_p(inc, limit_organelle, repeat, 'size', data_organelle, ctrl_lst, name, save_path)
    dis.plot_minus_ln_p(inc, limit_organelle, repeat, 'raw_int', data_organelle, ctrl_lst, name, save_path)
    dis.plot_minus_ln_p(inc, limit_organelle, repeat, 'circ', data_organelle, ctrl_lst, name, save_path)
    dis.plot_minus_ln_p(inc, limit_organelle, repeat, 'ecce', data_organelle, ctrl_lst, name, save_path)
    print("# Export organelle violin plots ...")
    dis.plot_violin('size', data_organelle, save_path, name)
    dis.plot_violin('raw_int', data_organelle, save_path, name)
    dis.plot_violin('circ', data_organelle, save_path, name)
    dis.plot_violin('ecce', data_organelle, save_path, name)

data_frap = pd.DataFrame({'sample': data_name, 'n_curve': data_frap_n_curve, 'limit': data_frap_phenotype_limit,
                          'mob': data_frap_phenotype_mob, 'curve_mob': data_frap_phenotype_curve_mob,
                          't_half': data_frap_phenotype_t_half, 'curve_t_half': data_frap_phenotype_curve_t_half,
                          'slope': data_frap_phenotype_slope, 'curve_slope': data_frap_phenotype_curve_slope,
                          'n_organelle': data_organelle_n, 'limit_organelle': data_organelle_phenotype_limit,
                          'size_organelle': data_organelle_phenotype_size,
                          'raw_int_organelle': data_organelle_phenotype_int,
                          'limit_organelle_circ': data_organelle_phenotype_limit_circ,
                          'circ_organelle': data_organelle_phenotype_circ,
                          'ecce_organelle': data_organelle_phenotype_ecce})

save_path = ("%ssummary/" % save_source)
if not os.path.exists(save_path):
    os.makedirs(save_path)

data_frap.to_csv('%ssummary.txt' % save_path, index=False, sep='\t')

print("DONE!")



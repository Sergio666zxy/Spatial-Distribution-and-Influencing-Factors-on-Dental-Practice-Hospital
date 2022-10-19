# %%
import numpy as np
import libpysal as ps
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mgwr.utils import shift_colormap, truncate_colormap

# %%


path1 = 'yourpath'
path2 = '.shp'

# %%
for i in range(0,5):
    filename = path1 + str(2016+i) + path2
    shp_df7 = gp.GeoDataFrame.from_file(filename, encoding='utf-8-sig')

    shp_df7.drop(columns = ['省', '市'], inplace = True)
    shp_df7.rename(columns = { '县': 'district', '乡': 'name'}, inplace = True)

# %%
    g_y = shp_df7['dentist'].values.reshape((-1,1))

    g_X = shp_df7[['nigh light', 'pop_mean', 'polu_mean', 'primary', 'secondary', 'child',
       'old_guy', 'supermarke']].values

    u = shp_df7.centroid.x
    v = shp_df7.centroid.y # need to extract the core's longitude and latitude
    g_coords = list(zip(u,v))

    g_X = (g_X - g_X.mean(axis=0)) / g_X.std(axis=0)
    g_y = (g_y - g_y.mean(axis=0)) / g_y.std(axis=0)

    # %%
    # 带宽选择函数
    gwr_selector = Sel_BW(g_coords, g_y, g_X)
    gwr_bw = gwr_selector.search(search_method='golden_section',criterion='AICc', max_iter = 500)
    print('最佳带宽大小为：',gwr_bw)

    selector = Sel_BW(g_coords, g_y, g_X, multi=True)
    selector.search(multi_bw_min=[2])

    # %%
    # GWR拟合

    gwr_results = GWR(g_coords, g_y, g_X, gwr_bw, fixed=False, kernel='bisquare', constant=True, spherical=True).fit()
    model = MGWR(g_coords, g_y, g_X, selector, fixed=False, kernel='bisquare', constant=True, spherical=True, sigma2_v1=True).fit()

    # %%
    gwr_results.summary()

    # %%
    model.summary()

    # %%
    var_names=['cof_Intercept', 'coef_nigh light', 'coef_pop_mean', 'coef_polu_mean', 'coef_primary', 'coef_secondary', 'coef_child',
           'coef_old_guy', 'coef_supermarke']
    gwr_coefficent = pd.DataFrame(model.params,columns=var_names)
    # 回归参数显著性
    gwr_flter_t = pd.DataFrame(model.filter_tvals(alpha = 0.9))

    # %%
    shp_df7.geom_type

    # %%
    # 将点数据回归结果放到面上展示
    # 主要是由于两个文件中的记录数不同，矢量面中的记录比csv中多几条，因此需要将没有参加gwr的区域去掉
    georgia_data_geo = gp.GeoDataFrame(shp_df7, geometry = gp.points_from_xy(shp_df7.centroid.x, shp_df7.centroid.y))
    georgia_data_geo = georgia_data_geo.join(gwr_coefficent)

    # %%
    shp_df7.geom_type

    # %%
    filename = path1 + str(2016+i) + path2
    shp_df7 = gp.GeoDataFrame.from_file(filename, encoding='utf-8-sig')
    shp_df7.rename(columns = {'index': 'region'}, inplace = True)
    georgia_shp_geo = gp.sjoin(shp_df7, georgia_data_geo, how="inner", op='contains').reset_index()

    # %%
    adjusted_varname = ['constant', 'nighttime light', 'population number', 'air pollution', 'primary school', 'middle school', 'kindergarten','almshouse', 'shopping']

    # %%
    fig,ax = plt.subplots(nrows=3, ncols=3, figsize = (50, 30))
    axes = ax.flatten()

    for i in range(0, len(axes)-1):

        ax=axes[i]
        cmap = plt.cm.brg # georgia_data_geo
        ax.set_title("MGWR: coefficient of " + var_names[i])

        mgwr_min = georgia_shp_geo[var_names[i]].min()
        mgwr_max = georgia_shp_geo[var_names[i]].max()
        #If all values are negative use the negative half of the colormap
        if (mgwr_max < 0) & (mgwr_max < 0):
            cmap = truncate_colormap(cmap, 0.0, 0.5)
        #If all values are positive use the positive half of the colormap
        elif (mgwr_max > 0) & (mgwr_max > 0): # georgia_shp_geo
            cmap = truncate_colormap(cmap, 0.5, 1.0)
        #Otherwise, there are positive and negative values so the colormap so zero is the midpoint
        else:
            cmap = shift_colormap(cmap, start=0.0, midpoint = 1 - mgwr_max/(mgwr_max + abs(mgwr_min)), stop=1.)

        # Create scalar mappable for colorbar and stretch colormap across range of data values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=mgwr_min, vmax=mgwr_max))


        georgia_shp_geo.plot(ax = ax,column=var_names[i], edgecolor='white', cmap=sm.cmap, vmin=mgwr_min, vmax=mgwr_max,legend=True, alpha = 0.6)

        #if (gwr_flter_t[i] == 0).any():
        #    georgia_shp_geo[gwr_flter_t[i] == 0].plot(color='lightgrey', ax=ax, edgecolor='white') # 灰色部分表示该系数不显著


        ax.set_axis_off()
        if i+1==8:
            axes[8].axis('off')

    plt.show()

# %%




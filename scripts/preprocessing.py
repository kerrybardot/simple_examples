import utils
import math
import flopy
import geopandas as gpd
import matplotlib.pyplot as plt

def make_well_gdf(df, vgrid, top, botm, crs = None):

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=crs)

    gdf['icpl'] = gdf.apply(lambda row: vgrid.intersect(row.x,row.y), axis=1)
    gdf['ground'] = gdf.apply(lambda row: top[row.icpl], axis=1)
    gdf['model_bottom'] = gdf.apply(lambda row: botm[-1, row.icpl], axis=1)
    gdf['z-bot'] = gdf.apply(lambda row: row['z'] - row['model_bottom'], axis=1)

    for idx, row in gdf.iterrows():
        result = row['z'] - row['model_bottom']
        if result < 0:
            print(f"Bore {row['id']} has an elevation below model bottom by: {result} m, removing from obs list")

    gdf = gdf[gdf['z-bot'] > 0] # filters out observations that are below the model bottom
    gdf['cell_disv'] = gdf.apply(lambda row: utils.xyz_to_disvcell(vgrid, row.x, row.y, row.z), axis=1)

    gdf['(lay,icpl)'] = gdf.apply(lambda row: utils.disvcell_to_layicpl(vgrid, row['cell_disv']), axis = 1)
    gdf['lay']        = gdf.apply(lambda row: row['(lay,icpl)'][0], axis = 1)
    gdf['icpl']       = gdf.apply(lambda row: row['(lay,icpl)'][1], axis = 1)
    gdf['wellcell_xy'] = gdf['icpl'].apply(lambda icpl: (vgrid.xcellcenters[icpl], vgrid.ycellcenters[icpl]))
    gdf['wellcell_z']  = gdf.apply(lambda row: vgrid.zcellcenters[row['lay'], row['icpl']], axis=1)
    gdf['well_zpillar']  = gdf.apply(lambda row: vgrid.zcellcenters[:, row['icpl']], axis=1)

    # Make sure no pinched out observations
    #if -1 in gdf['cell_disu'].values:
    #    print('Warning: some observations are pinched out. Check the model and data.')
    #    print('Number of pinched out observations removed: ', len(gdf[gdf['cell_disu'] == -1]))
    #    gdf = gdf[gdf['cell_disu'] != -1] # delete pilot points where layer is pinched out

    return gdf

def plot_array(vgrid, array, vmin = None, vmax = None, 
                levels = None, title = None, layer = 0,
                xlim = None, ylim = None, xy = None):
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    if title: ax.set_title(title)
    pmv = flopy.plot.PlotMapView(modelgrid=vgrid)
    t = pmv.plot_array(array, vmin = vmin, vmax = vmax)
    cbar = plt.colorbar(t, shrink = 0.5)  
    if xy:# Plot points
        ax.plot(xy[0], xy[1], 'o', ms = 2, color = 'red')
    if xlim: ax.set_xlim(xlim) 
    if ylim: ax.set_ylim(ylim) 

def make_obs_gdf(df, vgrid, crs = None):

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=crs)

    gdf['icpl'] = gdf.apply(lambda row: vgrid.intersect(row.x,row.y), axis=1)
    gdf['ground'] = gdf.apply(lambda row: vgrid.top[row.icpl], axis=1)
    gdf['model_bottom'] = gdf.apply(lambda row: vgrid.botm[-1, row.icpl], axis=1)
    gdf['z-bot'] = gdf.apply(lambda row: row['z'] - row['model_bottom'], axis=1)

    for idx, row in gdf.iterrows():
        result = row['z'] - row['model_bottom']
        if result < 0:
            print(f"Bore {row['id']} has an elevation below model bottom by: {result} m, removing from obs list")

    gdf = gdf[gdf['z-bot'] > 0] # filters out observations that are below the model bottom
    gdf['cell_disv'] = gdf.apply(lambda row: utils.xyz_to_disvcell(vgrid, row.x, row.y, row.z), axis=1)

    gdf['(lay,icpl)'] = gdf.apply(lambda row: utils.disvcell_to_layicpl(vgrid, row['cell_disv']), axis = 1)
    gdf['lay']        = gdf.apply(lambda row: row['(lay,icpl)'][0], axis = 1)
    gdf['icpl']       = gdf.apply(lambda row: row['(lay,icpl)'][1], axis = 1)
    gdf['obscell_xy'] = gdf['icpl'].apply(lambda icpl: (vgrid.xcellcenters[icpl], vgrid.ycellcenters[icpl]))
    gdf['obscell_z']  = gdf.apply(lambda row: vgrid.zcellcenters[row['lay'], row['icpl']], axis=1)
    gdf['obs_zpillar']  = gdf.apply(lambda row: vgrid.zcellcenters[:, row['icpl']], axis=1)

    # Make sure no pinched out observations
    #if -1 in gdf['cell_disu'].values:
    #    print('Warning: some observations are pinched out. Check the model and data.')
    #    print('Number of pinched out observations removed: ', len(gdf[gdf['cell_disu'] == -1]))
    #    gdf = gdf[gdf['cell_disu'] != -1] # delete pilot points where layer is pinched out

    return gdf

def make_obs_recarray(obs_gdf): ### CREATE REC ARRAY FOR MODFLOW
    
    xobs = obs_gdf.x.tolist()
    yobs = obs_gdf.y.tolist()
    zobs = obs_gdf.z.tolist()
    obslist = list(zip(xobs, yobs, zobs))

    # Create input arrays
    obscellid_list = obs_gdf.id.tolist()
    obscelllay_list = obs_gdf.lay.tolist()
    obscellicpl_list = obs_gdf.icpl.tolist()

    obs_rec = []
    for i in range(len(obscellid_list)):
        lay = obscelllay_list[i]
        icpl = obscellicpl_list[i]
        obs_rec.append([obscellid_list[i], 'head', (lay, icpl)]) 

    return obs_rec
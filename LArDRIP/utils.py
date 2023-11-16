import matplotlib.pyplot as plt
import numpy as np

def select_within_box(vox, data, bounds):
    keptVoxelMask = vox[:, 0] >= bounds[0][0]
    keptVoxelMask = np.logical_and(keptVoxelMask, vox[:, 0] < bounds[0][1])
    keptVoxelMask = np.logical_and(keptVoxelMask, vox[:, 1] >= bounds[1][0])
    keptVoxelMask = np.logical_and(keptVoxelMask, vox[:, 1] < bounds[1][1])
    keptVoxelMask = np.logical_and(keptVoxelMask, vox[:, 2] >= bounds[2][0])
    keptVoxelMask = np.logical_and(keptVoxelMask, vox[:, 2] < bounds[2][1])

    maskedVox = vox[keptVoxelMask]
    maskedData = data[keptVoxelMask]

    return maskedVox, maskedData

def draw_rect_bounds(ax, bounds, **kwargs):
    plotKwargs = {'ls': '--',
                  'color': 'red',
                  }
    plotKwargs.update(kwargs)
    
    ax.plot([bounds[0][0], bounds[0][1]],
            [bounds[1][0], bounds[1][0]],
            [bounds[2][0], bounds[2][0]],
            **plotKwargs
            )
    ax.plot([bounds[0][0], bounds[0][1]],
            [bounds[1][1], bounds[1][1]],
            [bounds[2][0], bounds[2][0]],
            **plotKwargs
            )
    ax.plot([bounds[0][0], bounds[0][1]],
            [bounds[1][0], bounds[1][0]],
            [bounds[2][1], bounds[2][1]],
            **plotKwargs
            )
    ax.plot([bounds[0][0], bounds[0][1]],
            [bounds[1][1], bounds[1][1]],
            [bounds[2][1], bounds[2][1]],
            **plotKwargs
            )
    
    ax.plot([bounds[0][0], bounds[0][0]],
            [bounds[1][0], bounds[1][1]],
            [bounds[2][0], bounds[2][0]],
            **plotKwargs
            )
    ax.plot([bounds[0][1], bounds[0][1]],
            [bounds[1][0], bounds[1][1]],
            [bounds[2][0], bounds[2][0]],
            **plotKwargs
            )
    ax.plot([bounds[0][0], bounds[0][0]],
            [bounds[1][0], bounds[1][1]],
            [bounds[2][1], bounds[2][1]],
            **plotKwargs
            )
    ax.plot([bounds[0][1], bounds[0][1]],
            [bounds[1][0], bounds[1][1]],
            [bounds[2][1], bounds[2][1]],
            **plotKwargs
            )

    ax.plot([bounds[0][0], bounds[0][0]],
            [bounds[1][0], bounds[1][0]],
            [bounds[2][0], bounds[2][1]],
            **plotKwargs
            )
    ax.plot([bounds[0][1], bounds[0][1]],
            [bounds[1][0], bounds[1][0]],
            [bounds[2][0], bounds[2][1]],
            **plotKwargs
            )
    ax.plot([bounds[0][0], bounds[0][0]],
            [bounds[1][1], bounds[1][1]],
            [bounds[2][0], bounds[2][1]],
            **plotKwargs
            )
    ax.plot([bounds[0][1], bounds[0][1]],
            [bounds[1][1], bounds[1][1]],
            [bounds[2][0], bounds[2][1]],
            **plotKwargs
            )

    return ax

def draw_patched_image(ax, patches, patchScheme, **kwargs):
    
    patchOffsets = np.array([[patchScheme[i]['xmin'],
                              patchScheme[i]['ymin'],
                              patchScheme[i]['zmin']]
                             for i in patches['patchInd']]).T
    patchPositions = np.array([patches['voxx'],
                               patches['voxy'],
                               patches['voxz']])
    patchPositions += patchOffsets

    ax.scatter(*patchPositions,
               **kwargs
               )

    ax.set_xlim(np.min(patchScheme['xmin']),
                np.max(patchScheme['xmax']))
    ax.set_ylim(np.min(patchScheme['ymin']),
                np.max(patchScheme['ymax']))
    ax.set_zlim(np.min(patchScheme['zmin']),
                np.max(patchScheme['zmax']))

    ax.legend(frameon = False)

    return ax

def draw_unpatched_image(ax, points, **kwargs):
    
    ax.scatter(*points,
               **kwargs
               )

    ax.set_xlim(np.min(patchScheme['xmin']),
                np.max(patchScheme['xmax']))
    ax.set_ylim(np.min(patchScheme['ymin']),
                np.max(patchScheme['ymax']))
    ax.set_zlim(np.min(patchScheme['zmin']),
                np.max(patchScheme['zmax']))

    ax.legend(frameon = False)

    return ax

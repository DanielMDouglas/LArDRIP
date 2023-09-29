import matplotlib.pyplot as plt
import numpy as np
import torch

def select_within_box(vox, data, bounds):
    keptVoxelMask = vox[:, 0] >= bounds[0][0]
    keptVoxelMask = np.logical_and(keptVoxelMask, vox[:, 0] < bounds[0][1])
    keptVoxelMask = np.logical_and(keptVoxelMask, vox[:, 1] >= bounds[1][0])
    keptVoxelMask = np.logical_and(keptVoxelMask, vox[:, 1] < bounds[1][1])

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
            **plotKwargs
            )
    ax.plot([bounds[0][0], bounds[0][1]],
            [bounds[1][1], bounds[1][1]],
            **plotKwargs
            )
    ax.plot([bounds[0][0], bounds[0][1]],
            [bounds[1][0], bounds[1][0]],
            **plotKwargs
            )
    ax.plot([bounds[0][0], bounds[0][1]],
            [bounds[1][1], bounds[1][1]],
            **plotKwargs
            )
    
    ax.plot([bounds[0][0], bounds[0][0]],
            [bounds[1][0], bounds[1][1]],
            **plotKwargs
            )
    ax.plot([bounds[0][1], bounds[0][1]],
            [bounds[1][0], bounds[1][1]],
            **plotKwargs
            )
    ax.plot([bounds[0][0], bounds[0][0]],
            [bounds[1][0], bounds[1][1]],
            **plotKwargs
            )
    ax.plot([bounds[0][1], bounds[0][1]],
            [bounds[1][0], bounds[1][1]],
            **plotKwargs
            )

    return ax

def draw_sparse_unpatched_image(ax, vox, data, **kwargs):

    ax.scatter(*vox.T,
               c = data.T,
               **kwargs
               )
        
    ax.legend(frameon = False)

    return ax

def draw_dense_image(ax, img, **kwargs):

    # ax.imshow(np.log(img),
    #           origin = 'lower',
    #           **kwargs
    #           )
    ax.imshow(img,
              # origin = 'lower',
              **kwargs
              )
        
    ax.legend(frameon = False)

    return ax

def draw_patched_image(ax, patches, patchScheme, unravelled = True, **kwargs):

    if unravelled:
        patches = ravel_patches(patches, patchScheme)
        
    patchPositions = np.array([patches['voxx'],
                               patches['voxy']])

    ax.scatter(*patchPositions,
               **kwargs
               )
        
    ax.set_xlim(np.min(patchScheme['xmin']),
                np.max(patchScheme['xmax']))
    ax.set_ylim(np.min(patchScheme['ymin']),
                np.max(patchScheme['ymax']))

    ax.legend(frameon = False)

    return ax

def ravel_patches(patches, patchScheme):
    
    patchOffsets = np.array([[patchScheme[i]['xmin'],
                              patchScheme[i]['ymin']]
                             for i in patches['patchInd']]).T
    patchPositions = np.array([patches['voxx'],
                               patches['voxy']])
    patchPositions += patchOffsets

    newPatches = patches.copy()
    newPatches['voxx'] = patchPositions[0]
    newPatches['voxy'] = patchPositions[1]

    return newPatches

def densify(patches, patchBounds, patchSizes):
    denseImage = np.zeros((len(patchBounds), patchSizes[0][0], patchSizes[0][1],))
    print (denseImage)
    for patchInd, patchBound in enumerate(patchBounds):
        for xi in range(patchSizes[0][0]):
            for yi in range(patchSizes[0][1]):
                mask = np.logical_and(np.logical_and(patches['voxx'] == xi,
                                                     patches['voxy'] == yi),
                                      patches['patchInd'] == patchInd)
                if any(mask):
                    denseImage[patchInd,xi,yi] = patches[mask]['voxq'][0]

    return patches

def plot_single_image_from_batch(model, imageBatch, out, mask, outfile):
    out = model.unpatchify(out)
    out = torch.einsum('nchw->nhwc', out).detach().cpu()

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * model.num_chans)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    image = torch.einsum('nchw->nhwc', imageBatch).cpu()

    im_masked = image * (1 - mask)

    im_paste = image * (1 - mask) + out * mask

    fig = plt.figure()
    fig.set_figheight(20)
    fig.set_figwidth(80)

    ax = fig.add_subplot(141)
    draw_dense_image(ax, image[0])

    ax = fig.add_subplot(142)
    draw_dense_image(ax, im_masked[0])

    ax = fig.add_subplot(143)
    draw_dense_image(ax, out[0])

    ax = fig.add_subplot(144)
    draw_dense_image(ax, im_paste[0])

    fig.savefig(outfile)

import h5py
import trimesh
import numpy as np
import matplotlib.cm as cm

with h5py.File("/path/to/h5_file.h5", 'r') as f:
    patched_data = f['patchedData']
    patch_bounds = f['patchBounds']
    
    # get length of image ids and patch ids
    num_images = len(np.unique(patched_data['imageInd']))
    num_patches = len(patch_bounds)
    
    for image_ind in range(num_images):
        if image_ind == 5:
            break
        # make an empty point cloud for each image
        vertices = []
        colors = []
        
        for patch_ind in range(num_patches):
            image_indices = np.where(patched_data['imageInd'] == image_ind)
            patch_indices = np.where(patched_data['patchInd'] == patch_ind)
            
            common_indices = np.intersect1d(image_indices, patch_indices)
            
            if len(common_indices) > 0:
                voxx = patched_data['voxx'][common_indices]
                voxy = patched_data['voxy'][common_indices]
                voxz = patched_data['voxz'][common_indices]
                voxq = patched_data['voxq'][common_indices]

                # patch bounds
                patch_info = patch_bounds[patch_ind]
                xmin, xmax, ymin, ymax, zmin, zmax = patch_info[1], patch_info[2], patch_info[3], patch_info[4], patch_info[5], patch_info[6]

                # coordinate shift
                global_x = xmin + voxx
                global_y = ymin + voxy
                global_z = zmin + voxz

                global_coords = np.column_stack((global_x, global_y, global_z))

                # voxq -> cmap
                colormap = cm.get_cmap('viridis')  
                color_values = colormap(voxq)

                vertices.append(global_coords)
                colors.append(color_values)

        # combine all arrays
        all_vertices = np.vstack(vertices)
        all_colors = np.vstack(colors)

        # create a point cloud
        point_cloud = trimesh.points.PointCloud(vertices=all_vertices, colors=all_colors)

        # save the point cloud as a .ply file for each image
        output_ply_file = f"/path/to/out_put_ply/{image_ind}.ply"
        point_cloud.export(output_ply_file, 'ply')
        print(f"Point cloud for imageInd {image_ind} saved to {output_ply_file}")

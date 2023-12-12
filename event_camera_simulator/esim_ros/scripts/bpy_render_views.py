"""
Adapted from `360_view.py` & `360_view_test.py` in the original NeRF synthetic
Blender dataset blend-files.
"""
import argparse
import os
import json
from math import radians
import bpy
import numpy as np


COLOR_SPACES = [ "display", "linear" ]
DEVICES = [ "cpu", "cuda", "optix" ]

CIRCLE_FIXED_START = ( 0, 0, 0 )
CIRCLE_FIXED_END = ( .7, 0, 0 )


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty

    return b_empty


def main(args):
    # open the scene blend-file
    bpy.ops.wm.open_mainfile(filepath=args.blend_path)

    # initialize render settings
    scene = bpy.data.scenes["Scene"]
    scene.render.engine = "CYCLES"
    scene.render.use_persistent_data = True

    if args.device == "cpu":
        bpy.context.preferences.addons["cycles"].preferences \
           .compute_device_type = "NONE"
        bpy.context.scene.cycles.device = "CPU"
    elif args.device == "cuda":
        bpy.context.preferences.addons["cycles"].preferences \
           .compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"
    elif args.device == "optix":
        bpy.context.preferences.addons["cycles"].preferences \
           .compute_device_type = "OPTIX"
        bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    # initialize compositing nodes
    scene.view_layers[0].use_pass_combined = True
    scene.use_nodes = True
    tree = scene.node_tree

    if args.depth:
        scene.view_layers[0].use_pass_z = True
        combine_color = tree.nodes.new("CompositorNodeCombineColor")
        depth_output = tree.nodes.new("CompositorNodeOutputFile")
    if args.normal:
        scene.view_layers[0].use_pass_normal = True
        normal_output = tree.nodes.new("CompositorNodeOutputFile")
    if args.depth or args.normal:
        render_layers = tree.nodes.new("CompositorNodeRLayers")

    # initialize RGB render image output settings
    scene.render.filepath = args.renders_path
    scene.render.use_file_extension = True
    scene.render.use_overwrite = True
    scene.render.image_settings.color_mode = "RGBA"

    if args.color_space == "display":
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_depth = "8"
        scene.render.image_settings.color_management = "FOLLOW_SCENE"
    elif args.color_space == "linear":
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.use_zbuffer = False

    if args.depth:
        # initialize depth render image output settings
        depth_output.base_path = os.path.join(args.renders_path, "depth")
        depth_output.file_slots[0].use_node_format = True
        scene.frame_set(0)

        depth_output.format.file_format = "OPEN_EXR"
        depth_output.format.color_mode = "RGB"
        depth_output.format.color_depth = "32"
        depth_output.format.exr_codec = "NONE"
        depth_output.format.use_zbuffer = False

        # link compositing nodes
        links = tree.links

        # output depth img (RGB img is output via the existing composite node)
        combine_color.mode = "RGB"
        links.new(render_layers.outputs["Depth"], combine_color.inputs["Red"])
        combine_color.inputs["Green"].default_value = 0
        combine_color.inputs["Blue"].default_value = 0
        combine_color.inputs["Alpha"].default_value = 1

        links.new(combine_color.outputs["Image"], depth_output.inputs["Image"])
    
    if args.normal:
        # initialize normal render image output settings
        normal_output.base_path = os.path.join(args.renders_path, "normal")
        normal_output.file_slots[0].use_node_format = True
        scene.frame_set(0)

        normal_output.format.file_format = "OPEN_EXR"
        normal_output.format.color_mode = "RGB"
        normal_output.format.color_depth = "32"
        normal_output.format.exr_codec = "NONE"
        normal_output.format.use_zbuffer = False

        # link compositing nodes
        links = tree.links

        # output normal img (RGB img is output via the existing composite node)
        combine_color.mode = "RGB"
        links.new(render_layers.outputs["Normal"],
                  normal_output.inputs["Image"])

    # initialize camera settings
    scene.render.dither_intensity = 0.0
    scene.render.film_transparent = True
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]

    cam = bpy.data.objects["Camera"]
    cam.location = (0, 4.0, 0.5)
    cam.rotation_mode = "XYZ"
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

    # preprocess & derive paths
    args.renders_path = os.path.normpath(args.renders_path)                     # remove trailing slashes
    folder_name = os.path.basename(args.renders_path)
    renders_parent_path = os.path.dirname(args.renders_path)  
    transforms_path = os.path.join(
        renders_parent_path, f"transforms_{folder_name}.json"
    )

    # render novel views
    stepsize = 360.0 / args.num_views
    if not args.random_views:
        vertical_diff = CIRCLE_FIXED_END[0] - CIRCLE_FIXED_START[0]
        b_empty.rotation_euler = CIRCLE_FIXED_START
        b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + vertical_diff

    out_data = {
        "camera_angle_x": cam.data.angle_x,
        "frames": []
    }
    for i in range(0, args.num_views):
        if args.random_views:
            if args.upper_views:
                rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
        else:
            print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
            
        scene.render.filepath = os.path.join(args.renders_path, f"r_{i}")
        if args.depth:
            depth_output.file_slots[0].path = f"r_{i}"
        if args.normal:
            normal_output.file_slots[0].path = f"r_{i}"
        bpy.ops.render.render(write_still=True)

        # remove the "0000" suffix in the depth & normal map filenames
        if args.depth:
            os.rename(os.path.join(depth_output.base_path, f"r_{i}0000.exr"),
                      os.path.join(depth_output.base_path, f"r_{i}.exr"))
        if args.normal:
            os.rename(os.path.join(normal_output.base_path, f"r_{i}0000.exr"),
                      os.path.join(normal_output.base_path, f"r_{i}.exr"))            

        frame_data = {
            "file_path": os.path.join(".", os.path.relpath(
                            scene.render.filepath, start=renders_parent_path
                         )),
            "rotation": radians(stepsize),
            "transform_matrix": listify_matrix(cam.matrix_world)
        }
        out_data["frames"].append(frame_data)

        if args.random_views:
            if args.upper_views:
                rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
        else:
            b_empty.rotation_euler[0] = (
                CIRCLE_FIXED_START[0]
                + (np.cos(radians(stepsize*i))+1)/2 * vertical_diff
            )
            b_empty.rotation_euler[2] += radians(2*stepsize)

    with open(transforms_path, "w") as out_file:
        json.dump(out_data, out_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Script for rendering novel views of"
                     " synthetic Blender scenes.")
    )
    parser.add_argument(
        "blend_path", type=str, default='/home/falcary/workstation/blender_env_indoors_dataset/AI58_001.blend',
        help="Path to the blend-file of the synthetic Blender scene."
    )
    parser.add_argument(
        "renders_path", type=str, default='/home/falcary/workstation/blender_env_indoors_dataset/outputs/AI58_001',
        help="Desired path to the novel view renders."
    )
    parser.add_argument(
        "num_views", type=int,
        help="Number of novel view renders."
    )
    parser.add_argument(
        "resolution", type=int, nargs=2, default=[1080,720],
        help="Image resolution of the novel view renders."
    )
    parser.add_argument(
        "--color_space", type=str, choices=COLOR_SPACES, default="display",
        help="Color space of the output novel view images."
    )
    parser.add_argument(
        "--device", type=str, choices=DEVICES, default="cuda",
        help="Compute device type for rendering."
    )
    parser.add_argument(
        "--random_views", action="store_true",
        help="Randomly sample novel views."
    )
    parser.add_argument(
        "--upper_views", action="store_true",
        help="Only sample novel views from the upper hemisphere."
    )
    parser.add_argument(
        "--depth", action="store_true",
        help="Render depth maps too."
    )
    parser.add_argument(
        "--normal", action="store_true",
        help="Render normal maps too."
    )
    args = parser.parse_args()

    main(args)

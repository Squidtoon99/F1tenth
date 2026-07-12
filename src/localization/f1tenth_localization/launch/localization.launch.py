"""On-car particle-filter localization (map_server + particle_filter).

Map and centerline paths default to the per-car overlay mounted at /config/maps.
Override via launch arguments when bench-testing with packaged assets.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    loc_share = get_package_share_directory('f1tenth_localization')
    default_map_yaml = '/config/maps/map.yaml'
    default_track_csv = '/config/maps/centerline.csv'
    default_pf_config = os.path.join(loc_share, 'config', 'pf_params.yaml')
    default_relocalize_config = os.path.join(loc_share, 'config', 'pf_relocalize.yaml')

    map_yaml = LaunchConfiguration('map_yaml')
    pf_config = LaunchConfiguration('pf_config')
    track_centerline_csv = LaunchConfiguration('track_centerline_csv')
    relocalize_config = LaunchConfiguration('relocalize_config')
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_relocalize = LaunchConfiguration('enable_relocalize')

    declare_map = DeclareLaunchArgument(
        'map_yaml',
        default_value=default_map_yaml,
        description='Occupancy grid YAML (per-car overlay).',
    )
    declare_pf = DeclareLaunchArgument(
        'pf_config',
        default_value=default_pf_config,
        description='Particle filter parameter YAML.',
    )
    declare_track = DeclareLaunchArgument(
        'track_centerline_csv',
        default_value=default_track_csv,
        description='Track centerline CSV in map frame (for track spread relocalize).',
    )
    declare_relocalize = DeclareLaunchArgument(
        'relocalize_config',
        default_value=default_relocalize_config,
        description='Joystick relocalize node parameters.',
    )
    declare_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock (true in gym).',
    )
    declare_enable_relocalize = DeclareLaunchArgument(
        'enable_relocalize',
        default_value='true',
        description='Start the PS4 relocalize helper node.',
    )

    lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': True},
            {'node_names': ['map_server']},
        ],
    )

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            {'yaml_filename': map_yaml},
            {'topic': 'map'},
            {'frame_id': 'map'},
            {'use_sim_time': use_sim_time},
        ],
    )

    pf_node = Node(
        package='particle_filter',
        executable='particle_filter',
        name='particle_filter',
        output='screen',
        parameters=[
            pf_config,
            {'track_centerline_csv': track_centerline_csv},
            {'use_sim_time': use_sim_time},
        ],
    )

    relocalize_node = Node(
        package='f1tenth_localization',
        executable='pf_relocalize',
        name='pf_relocalize',
        output='screen',
        parameters=[relocalize_config],
        condition=IfCondition(enable_relocalize),
    )

    return LaunchDescription([
        declare_map,
        declare_pf,
        declare_track,
        declare_relocalize,
        declare_sim_time,
        declare_enable_relocalize,
        lifecycle_node,
        map_server_node,
        pf_node,
        relocalize_node,
    ])

from dm_control import composer

# import mouse entity
from vnl_ray.mouse_forelimb.mouse_entity import MouseEntity
from vnl_ray.tasks.arenas.mouse_arena import MouseReachArena
from vnl_ray.tasks.mouse_reach_task import MouseReachTask
from dm_control import mjcf

_CONTROL_TIMESTEP = 0.005
_PHYSICS_TIMESTEP = 0.001


def mouse_reach(random_state=None, actuator_type=None):
    if actuator_type == "muscle":
        mouse_xml_path = "/root/vast/eric/vnl-ray/vnl_ray/mouse_forelimb/assets_mousereach/armmodel_atscale_working_balljoint_muscle.xml"
    elif actuator_type == "muscle_simple":
        mouse_xml_path = "/root/vast/eric/vnl-ray/vnl_ray/mouse_forelimb/assets_mousereach/akira_arm_model_v3.xml"
    elif actuator_type == "torque":
        mouse_xml_path = "/root/vast/eric/vnl-ray/vnl_ray/mouse_forelimb/assets_mousereach/akira_torque.xml"
    elif actuator_type == "position":
        mouse_xml_path = "/root/vast/eric/vnl-ray/vnl_ray/mouse_forelimb/assets_mousereach/armmodel_atscale_working_balljoint_position.xml"
    else:
        raise ValueError(f"Input Error: Input actuator type")

    # Initialize the MouseEntity without passing the physics object directly
    mouse = MouseEntity(mouse_xml_path)

    # Create the arena
    arena = MouseReachArena()

    # Create the task
    task = MouseReachTask(
        mouse=mouse,
        arena=arena,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP,
    )

    # Return the environment
    return composer.Environment(
        time_limit=3,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )

"""Utils for fly tasks."""

from typing import Sequence
import numpy as np
from dm_control.utils.transformations import quat_mul, quat_inv

from flybody.quaternions import rotate_vec_with_quat


def make_ghost_fly(walker, visible=True, visible_legs=True):
  """Create a 'ghost' fly to serve as a tracking target."""
  # Remove model elements.
  for tendon in walker.mjcf_model.find_all('tendon'):
    tendon.remove()
  for joint in walker.mjcf_model.find_all('joint'):
    joint.remove()
  for act in walker.mjcf_model.find_all('actuator'):
    act.remove()
  for sensor in walker.mjcf_model.find_all('sensor'):
    if sensor.tag == 'touch' or sensor.tag == 'force':
      sensor.remove()
  for exclude in walker.mjcf_model.find_all('contact'):
    exclude.remove()
  all_bodies = walker.mjcf_model.find_all('body')
  for body in all_bodies:
    if body.name and body.name.startswith('wing'):
      body.remove()
  for light in walker.mjcf_model.find_all('light'):
    light.remove()
  for camera in walker.mjcf_model.find_all('camera'):
    camera.remove()
  for site in walker.mjcf_model.find_all('site'):
    site.rgba = (0, 0, 0, 0)
  # Disable contacts, possibly make invisible.
  for geom in walker.mjcf_model.find_all('geom'):
    # alpha=0.999 ensures grey ghost reference.
    # for alpha=1.0 there is no visible difference between real walker and
    # ghost reference.
    if not visible_legs and any_substr_in_str(['coxa', 'femur', 'tibia',
                                               'tarsus', 'claw'], geom.name):
      rgba = (0, 0, 0, 0)
    else:
      rgba = (0.5, 0.5, 0.5, 0.2 if visible else 0.0)
    geom.set_attributes(
      user=(0,),
      contype=0,
      conaffinity=0,
      rgba=rgba)
    if geom.mesh is None:
      geom.remove()


def retract_wings(physics: 'mjcf.Physics', prefix: str = 'walker',
                  roll=0.7, pitch=-1.0, yaw=1.5) -> None:
  """Set wing qpos to default retracted position."""
  for side in ['left', 'right']:
    physics.named.data.qpos[f'{prefix}/wing_roll_{side}'] = roll
    physics.named.data.qpos[f'{prefix}/wing_pitch_{side}'] = pitch
    physics.named.data.qpos[f'{prefix}/wing_yaw_{side}'] = yaw


def add_trajectory_sites(root_entity, n_traj_sites, group=4):
  """Adds trajectory sites to root entity."""
  for i in range(n_traj_sites):
    root_entity.mjcf_model.worldbody.add(
      element_name='site',
      name=f'traj_{i}',
      size=(0.005, 0.005, 0.005),
      rgba=(0, 1, 1, 0.5),
      group=group)


def update_trajectory_sites(root_entity, ref_qpos, n_traj_sites,
                            traj_timesteps):
  """Updates trajectory sites"""
  for i in range(n_traj_sites):
    site = root_entity.mjcf_model.worldbody.find('site', f'traj_{i}')
    if i < traj_timesteps // 10:
      site.pos = ref_qpos[10*i, :3]
      site.rgba[3] = 0.5
    else:
      # Hide extra sites beyond current trajectory length, if any.
      site.rgba[3] = 0.


def neg_quat(quat_a):
  """Returns neg(quat_a)."""
  quat_b = quat_a.copy()
  quat_b[0] *= -1
  return quat_b


def any_substr_in_str(substrings: Sequence[str], string: str) -> bool:
  """Checks if any of substrings is in string."""
  return any(s in string for s in substrings)


def qpos_name2id(physics: 'mjcf.Physics') -> dict:
    """Mapping from qpos (joint) names to qpos ids.
    Returns dict of `joint_name: [id(s)]` for physics.data.qpos."""
    name2id_map = {}
    idx = 0
    for j in range(physics.model.njnt):
      joint_name = physics.model.id2name(j, 'joint')
      qpos_slice = physics.named.data.qpos[joint_name]
      name2id_map[joint_name] = [*range(idx, idx+len(qpos_slice))]
      idx += len(qpos_slice)
    return name2id_map


def root2com(root_qpos, offset=np.array([-0.03697732, 0.00029205, -0.0142447])):
    """Get fly CoM in world coordinates using fixed offset from fly's
    root joint.

    This function is inverse of com2root.

    Args:
        root_qpos: qpos of root joint (pos & quat) in world coordinates, (7,).
        offset: CoM's offset from root in local thorax coordinates.

    Returns:
        CoM position in world coordinates, (3,).
    """
    offset_global = rotate_vec_with_quat(offset, root_qpos[3:])
    com = root_qpos[:3] + offset_global
    return com


def com2root(com, quat, offset=np.array([-0.03697732, 0.00029205, -0.0142447])):
    """Get position of fly's root joint from CoM position in global coordinates.

    This function is inverse of root2com.

    Any number of batch dimensions is supported.

    Args:
        com: CoM position in world coordinates, (B, 3,).
        quat: Orientation quaternioin of the fly, (B, 4,).
        offset: Offset from root joint to fly's CoM in local thorax coordinates.

    Returns:
        Position of fly's root joint is world coordinates, (B, 3,).
    """
    offset_global = rotate_vec_with_quat(-offset, quat)
    root_pos = com + offset_global
    return root_pos

## Steps to generate the MuJoCo fruit fly model

1. Starting point:
   * In `assets/`:
       * Raw fly model consisting of `drosophila.xml` and `.msh` meshes, as [exported](https://github.com/google-deepmind/dm_control/tree/main/dm_control/blender/mujoco_exporter) from [Blender](https://www.blender.org/).
       * Defaults file `drosophila_defaults.xml`.
    * Meshes [converted](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/msh2obj.py) from `.msh` to `.obj` format, located at `flybody/fruitfly/assets`. This is the "official" directory for the final fly MJCF we generate here.
3. Run `fuse_fruitfly.py`, it will create `assets/drosophila_fused.xml`.
4. Run `make_fruitfly.py`, it will create `fruitfly.xml`, a final fly model MJCF.
5. Copy the generated `fruitfly.xml` over to `flybody/fruitfly/assets` where the expected `.obj` meshes are.
6. Your fruit fly `flybody/fruitfly/assets/fruitfly.xml` is ready to go!
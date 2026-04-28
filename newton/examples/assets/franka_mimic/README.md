# Factory `franka_mimic.usd`

This directory is the expected location of `franka_mimic.usd`, the Panda
asset that IsaacLab's Factory environment uses for the
`Isaac-Factory-NutThread-Direct-v0` task. The file is **not** included in
the Newton repository because of upstream licensing on the Isaac Sim
asset bundle.

The compare viewer (`newton/examples/robot/example_robot_panda_compare.py`)
loads it as Robot B by default to give a side-by-side visual comparison
against the Newton URDF. If the file is missing, the example falls back
to a second copy of the Newton URDF and prints instructions.

## Where to obtain it

If you have IsaacLab installed, it pulls the asset from the Isaac
Nucleus and caches it locally. A typical path is:

```
${ISAAC_NUCLEUS_DIR}/IsaacLab/Factory/franka_mimic.usd
```

On a machine that has run IsaacLab once, the cached copy is usually at:

```
/tmp/Assets/Isaac/<version>/Isaac/IsaacLab/Factory/franka_mimic.usd
```

## How to install it

Copy the file into this directory:

```bash
cp /tmp/Assets/Isaac/6.0/Isaac/IsaacLab/Factory/franka_mimic.usd \
   newton/examples/assets/franka_mimic/franka_mimic.usd
```

Adjust the source path to wherever your Isaac Sim/IsaacLab installation
keeps it. Once the file is present at the expected path, the compare
viewer picks it up automatically.

## License

`franka_mimic.usd` is part of the Isaac Sim asset bundle, distributed
under NVIDIA's terms separately from this repository's open-source
license. Do **not** commit the file here.

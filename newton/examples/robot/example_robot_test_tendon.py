import newton
from newton.solvers import SolverMuJoCo


def test_basic_tendon():
    """Test that tendons work"""
    mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
      <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom solmix="1.0" type="cylinder" size="0.05 0.025" rgba="1 0 0 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>
      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom type="cylinder" size="0.05 0.025" rgba="0 0 1 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <!-- Fixed tendon coupling joint1 and joint2 -->
    <fixed
        name="coupling_tendon"
        stiffness="1"
        damping="2"
        
        >
      <joint joint="joint1" coef="1"/>
      <joint joint="joint2" coef="-1"/>
    </fixed>
    <!-- Fixed tendon coupling joint1 and joint2 -->
    <fixed
        name="coupling_tendon_reversed"
        stiffness="4"
        damping="5"
        
        >
      <joint joint="joint1" coef="1"/>
      <joint joint="joint2" coef="1"/>
    </fixed>
  </tendon>
</mujoco>
"""
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(mjcf)

    tendon_joint_adr = builder.custom_attributes["mujoco:tendon_joint_adr"].values or {}
    tendon_joint_vals = builder.custom_attributes["mujoco:tendon_joint"].values or {}
    print(f"Tendon joint adr: {tendon_joint_adr}")
    print(f"Tendon joint vals: {tendon_joint_vals}")
    print(f"Num tendons: {len(tendon_joint_adr)}")

    stiffness = builder.custom_attributes["mujoco:tendon_stiffness"].values or {}
    print(f"Stiffness: {stiffness}")

    damping = builder.custom_attributes["mujoco:tendon_damping"].values or {}
    print(f"Damping: {damping}")

    # TODO: springlength probably needs special handling.
    # See: https://mujoco.readthedocs.io/en/stable/XMLreference.html#tendon-spatial-springlength
    # springlength = builder.custom_attributes["mujoco:tendon_springlength"].values or {}
    # print(f"Spring length: {springlength}")

    model = builder.finalize()

    solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)
    print("Done")


if __name__ == "__main__":
    test_basic_tendon()

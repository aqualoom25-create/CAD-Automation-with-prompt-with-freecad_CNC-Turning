import cadquery as cq
import math

# Synthesized by: Dynamic Image-Driven 3D CSG Synthesizer (CadQuery OpenCASCADE B-Rep + Manifold3D CSG + SolidPython2 CSG)
# Part: Dynamic Image-Driven 3D CSG Machined Component
def build_vision_neural_csg_part(tol_offset_mm: float = 0.0) -> cq.Workplane:
    solid = None
    # Outer Feature 1: Z=[0.00, 9.75] mm, OD=9.48->30.21 mm (taper)
    s_0 = cq.Workplane('XZ').moveTo(0, 0.00).lineTo(max(0.5, (9.48+tol_offset_mm)/2.0), 0.00).lineTo(max(0.5, (30.21+tol_offset_mm)/2.0), 9.75).lineTo(0, 9.75).close().revolve(360, (0,0,0), (0,1,0))
    solid = s_0 if solid is None else solid.union(s_0)
    # Outer Feature 2: Z=[9.75, 42.00] mm, OD=35.84->24.87 mm (taper)
    s_1 = cq.Workplane('XZ').moveTo(0, 9.75).lineTo(max(0.5, (35.84+tol_offset_mm)/2.0), 9.75).lineTo(max(0.5, (24.87+tol_offset_mm)/2.0), 42.00).lineTo(0, 42.00).close().revolve(360, (0,0,0), (0,1,0))
    solid = s_1 if solid is None else solid.union(s_1)
    # Internal Bore / Counterbore Step 1: Z=[0.00, 42.00], ID=15.05 mm
    bore_0 = cq.Workplane('XY').workplane(offset=-0.20).circle(max(0.4, (15.05-tol_offset_mm)/2.0)).extrude(42.40)
    solid = solid.cut(bore_0)
    return solid

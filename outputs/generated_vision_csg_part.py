import cadquery as cq
import math

# Synthesized by: Local PyTorch Vision Neural Network + 3D CSG Boolean Synthesizer
# Part: CNC Mill-Turn Castellated & Side-Slotted Hollow Cylindrical Housing Sleeve
def build_vision_neural_csg_part(tol_offset_mm: float = 0.0) -> cq.Workplane:
    solid = None
    # Outer Feature 1: Z=[0.00, 1.20] mm, OD=40.90->42.50 mm (chamfer)
    s_0 = cq.Workplane('XZ').moveTo(0, 0.00).lineTo(max(0.5, (40.90+tol_offset_mm)/2.0), 0.00).lineTo(max(0.5, (42.50+tol_offset_mm)/2.0), 1.20).lineTo(0, 1.20).close().revolve(360, (0,0,0), (0,1,0))
    solid = s_0 if solid is None else solid.union(s_0)
    # Outer Feature 2: Z=[1.20, 34.20] mm, OD=42.50->42.50 mm (cylinder)
    s_1 = cq.Workplane('XY').workplane(offset=1.20).circle(max(0.5, (42.50 + tol_offset_mm)/2.0)).extrude(33.00)
    solid = s_1 if solid is None else solid.union(s_1)
    # Outer Feature 3: Z=[34.20, 40.40] mm, OD=42.50->42.50 mm (castellated_rim)
    s_2 = cq.Workplane('XY').workplane(offset=34.20).circle(max(0.5, (42.50 + tol_offset_mm)/2.0)).extrude(6.20)
    solid = s_2 if solid is None else solid.union(s_2)
    # Internal Bore / Counterbore Step 1: Z=[0.00, 23.03], ID=29.75 mm
    bore_0 = cq.Workplane('XY').workplane(offset=-0.20).circle(max(0.4, (29.75-tol_offset_mm)/2.0)).extrude(23.43)
    solid = solid.cut(bore_0)
    # Internal Bore / Counterbore Step 2: Z=[23.03, 40.40], ID=35.15 mm
    bore_1 = cq.Workplane('XY').workplane(offset=22.83).circle(max(0.4, (35.15-tol_offset_mm)/2.0)).extrude(17.77)
    solid = solid.cut(bore_1)
    # 3x Circumferential Side Window Cutouts (11.05 x 18.58 mm)
    return solid

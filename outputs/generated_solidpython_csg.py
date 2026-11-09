import solid2 as sp2
import math

# SolidPython2 + Manifold3D 3D CSG Boolean Tree for: Dynamic Image-Driven 3D CSG Machined Component
def build_solidpython_csg():
    unions = []
    unions.append(sp2.cylinder(r1=4.74, r2=15.105, h=9.75, _fn=96).up(0.00))
    unions.append(sp2.cylinder(r1=17.92, r2=12.435, h=32.25, _fn=96).up(9.75))
    body = sp2.union()(*unions)
    body -= sp2.cylinder(r=7.525, h=43.00, _fn=72).up(-0.50)
    return body

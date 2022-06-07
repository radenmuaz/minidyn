
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.linalg import Vector3, Matrix4, Quaternion
global cube
class View:
    "View the world"
    def __init__(self, world):
        self.canvas = WgpuCanvas()
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.camera = gfx.PerspectiveCamera(70, 16 / 9)
        self.camera.position.z = 400
        self.world = world

    def animate(self):
        global cube
        print('yes')
        rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
        cube.rotation.multiply(rot)

        self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw()

        
        

if __name__ == "__main__":
    view = View(None)
    canvas = view.canvas
    renderer = view.renderer
    scene = view.scene

    ground = gfx.Mesh(
        gfx.box_geometry(500, 10, 500),
        gfx.MeshPhongMaterial(color="#FFFFFF"),
    )
    ground.position = Vector3(0,-100,0)
    scene.add(ground)
    camera = view.camera
    cube = gfx.Mesh(
        gfx.box_geometry(100, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )
    scene.add(cube)
    camera = view.camera

 
    # canvas.request_draw(update)
    canvas.request_draw(view.animate)

    run()

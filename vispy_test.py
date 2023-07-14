<<<<<<< Updated upstream
import sys
import itertools

import numpy as np
from vispy import app, gloo, visuals
from vispy.visuals.transforms import NullTransform

from montecarlo.layers_n.with_numba import DipoleSim


class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, title="Quiver plot", keys="interactive",
                            size=(960, 830))

        self.arrow_length = 5

        self.grid_coords = None
        self.line_vertices = None
        self.last_mouse = (0, 0)

        self.generate_grid()

        self.visual = visuals.ArrowVisual(
            color='white',
            connect='segments',
            arrow_size=15
        )

        self.visual.events.update.connect(lambda evt: self.update())
        self.visual.transform = NullTransform()

        self.show()

    def generate_grid(self):
        num_cols = 30
        num_rows = 30
        a = self.physical_size[0] / num_cols * 0.5
        r = DipoleSim.gen_lattice_triangular_square(num_rows, num_cols)

        self.grid_coords = (r + 0.25) * a

    def on_resize(self, event):
        self.generate_grid()
        self.rotate_arrows(np.array(self.last_mouse))

        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.visual.transforms.configure(canvas=self, viewport=vp)

    def rotate_arrows(self, point_towards):
        direction_vectors = (self.grid_coords - point_towards).astype(
            np.float32)
        norms = np.sqrt(np.sum(direction_vectors ** 2, axis=-1))
        direction_vectors[:, 0] /= norms
        direction_vectors[:, 1] /= norms

        vertices = np.repeat(self.grid_coords, 2, axis=0)
        vertices[::2] = vertices[::2] + ((0.5 * self.arrow_length) *
                                         direction_vectors)
        vertices[1::2] = vertices[1::2] - ((0.5 * self.arrow_length) *
                                           direction_vectors)

        self.visual.set_data(
            pos=vertices,
            arrows=vertices.reshape((len(vertices) // 2, 4)),
        )

    def on_mouse_move(self, event):
        self.last_mouse = event.pos
        self.rotate_arrows(np.array(event.pos))

    def on_draw(self, event):
        gloo.clear('black')
        self.visual.draw()


if __name__ == '__main__':
    win = Canvas()

    if sys.flags.interactive != 1:
        app.run()
=======
import vispy as vp
import numpy as np

data = np.zeros(4, dtype=[("position", np.float32, 2),
                          ("color", np.float32, 4)])
>>>>>>> Stashed changes

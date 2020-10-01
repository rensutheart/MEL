import pyglet
import glooey
from pyglet.gl import GLubyte
import numpy as np

class MyLabel(glooey.Label):
    custom_color = '#ffffff'
    custom_font_size = 14
    custom_alignment = 'center'


class MyButton(glooey.Button):
    Foreground = MyLabel
    custom_alignment = 'fill'


    class Base(glooey.Background):
        custom_color = '#204a87'

    class Over(glooey.Background):
        custom_color = '#3465a4'

    class Down(glooey.Background):
        custom_color = '#729fcff'


    def __init__(self, text, response, height, on_click):
        super().__init__(text)
        self.response = response
        self.set_height_hint(height)
        self.on_click = on_click

    # def on_click(self, widget):
    #     print(self.response)


class ImageWidget(glooey.Widget):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.sprite = None
        self.set_alignment('center')
        self.imFlat = np.swapaxes(self.image, 0, 1)
        self.imFlat = self.image.flatten()

        self.imFlat = (GLubyte * len(self.imFlat))(*self.imFlat)


    def do_claim(self):
        return self.image.shape[1], self.image.shape[0]

    # Glooey calls this method when the widget is assigned a new group.
    # See the section on `How regrouping works` for more details.
    def do_regroup(self):
        if self.sprite is not None:
            self.sprite.batch = self.batch
            self.sprite.group = self.group

    def do_draw(self):
        if self.sprite is None:
            self.sprite = pyglet.sprite.Sprite(
                img=pyglet.image.ImageData(self.image.shape[1], self.image.shape[0], "RGB", self.imFlat),
                x=self.rect.left,
                y=self.rect.bottom,
                batch=self.batch,
                group=self.group,
            )



    def do_undraw(self):
        if self.sprite is not None:
            self.sprite.delete()
            self.sprite = None

    def update_image(self, image):
        self.image = image
        self.imFlat = np.swapaxes(self.image, 0, 1)
        self.imFlat = self.image.flatten()

        self.imFlat = (GLubyte * len(self.imFlat))(*self.imFlat)

        self.sprite = pyglet.sprite.Sprite(
            img=pyglet.image.ImageData(self.image.shape[1], self.image.shape[0], "RGB", self.imFlat),
            x=self.rect.left,
            y=self.rect.bottom,
            batch=self.batch,
            group=self.group,
        )

        self.set_alignment('center')



import os
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
from kivy.app import App
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.checkbox import CheckBox
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty, BooleanProperty, StringProperty


Window.size = (1000, 800)
gui_spacing = 20
sider_min = 0
sider_max = 2
sider_default = 1


def edit_photo(image_path, output_path, brightness_factor=1.0, contrast_factor=1.0, saturation_factor=1.0, sharpness_factor=1.0, sharpness_enhancement=False, color_balance=(1.0, 1.0, 1.0)):
    image = Image.open(image_path)
    exif = image.getexif()

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation_factor)

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness_factor)

    if sharpness_enhancement:
        image = image.filter(ImageFilter.UnsharpMask())

    r, g, b = image.split()
    r = r.point(lambda i: i * color_balance[0])
    g = g.point(lambda i: i * color_balance[1])
    b = b.point(lambda i: i * color_balance[2])
    image = Image.merge('RGB', (r, g, b))

    image.save(output_path, exif=exif)


def get_files(folder, include_subfolders, extension):
    files = []
    for root, dirs, all_files in os.walk(folder):
        for file in all_files:
            if file.endswith(extension):
                files.append(os.path.join(root, file))
        if not include_subfolders:
            break
    return files


class PhotoEditorApp(App):
    folder_value = StringProperty(r"C:\Users\eddy.a\Downloads\Takeout\Google Photos")
    include_subfolders_value = BooleanProperty(False)
    extension_value = StringProperty(".jpg")
    file_out_value = StringProperty("edit 01")
    export_log_value = BooleanProperty(True)

    brightness_value = NumericProperty(1)
    contrast_value = NumericProperty(1)
    saturation_value = NumericProperty(1)
    sharpness_value = NumericProperty(1)
    sharpness_enhancement_value = BooleanProperty(False)
    color_balance_r_value = NumericProperty(1)
    color_balance_g_value = NumericProperty(1)
    color_balance_b_value = NumericProperty(1)

    def build(self):
        main_layout = BoxLayout(orientation='vertical', spacing=gui_spacing)
        sliders_layout = BoxLayout(orientation='vertical', spacing=gui_spacing)

        # Inputs
        input_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, 0.25))
        input_layout.add_widget(Label(text="Folder"))
        self.folder_input = TextInput(text=self.folder_value)
        input_layout.add_widget(self.folder_input)
        main_layout.add_widget(input_layout)

        input_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, 0.01))
        input_layout.add_widget(Label(text="Include subfolders"))
        self.include_subfolders_input = CheckBox(active=self.include_subfolders_value)
        input_layout.add_widget(self.include_subfolders_input)
        main_layout.add_widget(input_layout)

        input_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, 0.1))
        input_layout.add_widget(Label(text="Extension to filter"))
        self.extension_input = TextInput(text=self.extension_value)
        input_layout.add_widget(self.extension_input)
        main_layout.add_widget(input_layout)

        input_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, 0.1))
        input_layout.add_widget(Label(text="Edit Name"))
        self.file_out_input = TextInput(text=self.file_out_value)
        input_layout.add_widget(self.file_out_input)
        main_layout.add_widget(input_layout)

        input_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, 0.01))
        input_layout.add_widget(Label(text="Export log"))
        self.export_log_input = CheckBox(active=self.export_log_value)
        input_layout.add_widget(self.export_log_input)
        main_layout.add_widget(input_layout)

        # Sliders
        slider_and_label_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing)
        self.brightness_label = Label(text=f"Brightness = {self.brightness_value:.2f}")
        slider_and_label_layout.add_widget(self.brightness_label)
        self.brightness_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.brightness_slider.bind(value=self.on_brightness_slider_change)
        slider_and_label_layout.add_widget(self.brightness_slider)
        sliders_layout.add_widget(slider_and_label_layout)

        slider_and_label_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing)
        self.contrast_label = Label(text=f"Contrast = {self.contrast_value:.2f}")
        slider_and_label_layout.add_widget(self.contrast_label)
        self.contrast_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.contrast_slider.bind(value=self.on_contrast_slider_change)
        slider_and_label_layout.add_widget(self.contrast_slider)
        sliders_layout.add_widget(slider_and_label_layout)

        slider_and_label_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing)
        self.saturation_label = Label(text=f"Saturation = {self.contrast_value:.2f}")
        slider_and_label_layout.add_widget(self.saturation_label)
        self.saturation_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.saturation_slider.bind(value=self.on_saturation_slider_change)
        slider_and_label_layout.add_widget(self.saturation_slider)
        sliders_layout.add_widget(slider_and_label_layout)

        slider_and_label_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing)
        self.sharpness_label = Label(text=f"Sharpness = {self.contrast_value:.2f}")
        slider_and_label_layout.add_widget(self.sharpness_label)
        self.sharpness_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.sharpness_slider.bind(value=self.on_sharpness_slider_change)
        slider_and_label_layout.add_widget(self.sharpness_slider)
        sliders_layout.add_widget(slider_and_label_layout)

        slider_and_label_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing)
        self.sharpness_enhancement_label = Label(text=f"Sharpness enhancement = {self.sharpness_enhancement_value}")
        slider_and_label_layout.add_widget(self.sharpness_enhancement_label)
        self.sharpness_enhancement_bool = CheckBox()
        self.sharpness_enhancement_bool.bind(active=self.on_sharpness_enhancement_bool_change)
        slider_and_label_layout.add_widget(self.sharpness_enhancement_bool)
        sliders_layout.add_widget(slider_and_label_layout)

        slider_and_label_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing)
        self.color_balance_r_label = Label(text=f"Color Balance (Red) = {self.color_balance_r_value:.2f}")
        slider_and_label_layout.add_widget(self.color_balance_r_label)
        self.color_balance_r_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.color_balance_r_slider.bind(value=self.on_color_balance_r_slider_change)
        slider_and_label_layout.add_widget(self.color_balance_r_slider)
        sliders_layout.add_widget(slider_and_label_layout)

        slider_and_label_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing)
        self.color_balance_g_label = Label(text=f"Color Balance (Green) = {self.color_balance_g_value:.2f}")
        slider_and_label_layout.add_widget(self.color_balance_g_label)
        self.color_balance_g_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.color_balance_g_slider.bind(value=self.on_color_balance_g_slider_change)
        slider_and_label_layout.add_widget(self.color_balance_g_slider)
        sliders_layout.add_widget(slider_and_label_layout)

        slider_and_label_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing)
        self.color_balance_b_label = Label(text=f"Color Balance (Blue) = {self.color_balance_b_value:.2f}")
        slider_and_label_layout.add_widget(self.color_balance_b_label)
        self.color_balance_b_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.color_balance_b_slider.bind(value=self.on_color_balance_b_slider_change)
        slider_and_label_layout.add_widget(self.color_balance_b_slider)
        sliders_layout.add_widget(slider_and_label_layout)

        edit_button = Button(text="Edit Image", size_hint=(0.25, 0.1), pos_hint={"x": 0.5 - (0.25 / 2)})
        edit_button.bind(on_press=self.start)
        main_layout.add_widget(sliders_layout)
        main_layout.add_widget(edit_button)
        return main_layout

    def on_brightness_slider_change(self, instance, value):
        self.brightness_value = value
        self.brightness_label.text = f"Brightness = {value:.2f}"

    def on_contrast_slider_change(self, instance, value):
        self.contrast_value = value
        self.contrast_label.text = f"Contrast = {value:.2f}"

    def on_saturation_slider_change(self, instance, value):
        self.saturation_value = value
        self.saturation_label.text = f"Saturation = {value:.2f}"

    def on_sharpness_slider_change(self, instance, value):
        self.sharpness_value = value
        self.sharpness_label.text = f"Sharpness = {value:.2f}"

    def on_sharpness_enhancement_bool_change(self, instance, value):
        self.sharpness_enhancement_value = value
        self.sharpness_enhancement_label.text = f"Sharpness enhancement = {value}"

    def on_color_balance_r_slider_change(self, instance, value):
        self.color_balance_r_value = value
        self.color_balance_r_label.text = f"Color Balance (Red) = {value:.2f}"

    def on_color_balance_g_slider_change(self, instance, value):
        self.color_balance_g_value = value
        self.color_balance_g_label.text = f"Color Balance (Green) = {value:.2f}"

    def on_color_balance_b_slider_change(self, instance, value):
        self.color_balance_b_value = value
        self.color_balance_b_label.text = f"Color Balance (Blue) = {value:.2f}"

    def start(self, instance):
        folder = self.folder_input.text
        include_subfolders = self.include_subfolders_input.active
        extension = self.extension_input.text
        edit_name = self.file_out_input.text
        file_out = f" ({edit_name})."

        brightness_factor = self.brightness_value
        contrast_factor = self.contrast_value
        saturation_factor = self.saturation_value
        sharpness_factor = self.sharpness_value
        sharpness_enhancement = self.sharpness_enhancement_value
        color_balance = (self.color_balance_r_value, self.color_balance_g_value, self.color_balance_b_value)

        string_applied_enhancements = f"""These are the applied enhancements:
{brightness_factor = :.2f}
{contrast_factor = :.2f}
{saturation_factor = :.2f}
{sharpness_factor = :.2f}
{sharpness_enhancement = :.2f}
color_balance = ({color_balance[0]:.2f}, {color_balance[1]:.2f}, {color_balance[2]:.2f})"""

        try:
            files = get_files(folder, include_subfolders, extension)
            if len(files) > 0:
                for index_file, file in enumerate(files):
                    edit_photo(file, file_out.join(file.rsplit('.', 1)), brightness_factor, contrast_factor, saturation_factor, sharpness_factor, sharpness_enhancement, color_balance)
            else:
                raise Exception("There are no files to edit")

            if self.export_log_value:
                f = open(f"{folder}\\Python Image Editor - {edit_name} ({datetime.now().strftime("%Y-%m-%d %H-%M-%S")}).txt", "a")
                f.write("Python Image Editor\n\n")
                f.write(f"{edit_name = }\n")
                f.write(f"Date = {datetime.now().strftime("%Y-%m-%d %H-%M-%S")}\n")
                f.write("Files:\n")
                for index_file, file in enumerate(files):
                    f.write(f"File {index_file + 1:03} = {file}\n")
                f.write(f"\n{string_applied_enhancements}\n")
                f.close()

            self.show_popup(f"Successfully edited {len(files)} file(s) with extension '{extension}' in folder:\n'{folder}'", string_applied_enhancements)
        except Exception as e:
            self.show_popup("Error", f"An error occurred: {str(e)}")

    @staticmethod
    def show_popup(title, content):
        popup = Popup(title=title, content=Label(text=content), size_hint=(None, None), size=(400, 400))
        popup.open()


if __name__ == "__main__":
    PhotoEditorApp().run()
